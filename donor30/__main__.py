"""Local lane entrypoint; never starts a different server remotely."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from . import plan


def main(argv=None):
    parser = argparse.ArgumentParser(description='30h: two fresh local donors -> frozen-donor Stage 2')
    sub = parser.add_subparsers(dest='command', required=True)
    smoke = sub.add_parser('smoke', help='Synthetic CPU tests only; no data or Google writes')
    smoke.add_argument('--verbose', action='store_true')
    for name in ('run', 'catalog', 'publish', 'publish-auto'):
        p = sub.add_parser(name)
        p.add_argument('--server', required=True, choices=plan.LANES)
        p.add_argument('--output', default='work_dir/donor30')
        if name == 'run':
            p.add_argument('--sources', default='ablr2/sensor_sources.json')
            p.add_argument('--device', default='cuda')
        if name in ('publish', 'publish-auto'):
            p.add_argument('--root')
            p.add_argument('--credentials', default='gspread/account.json')
            p.add_argument('--legacy-root', action='append', dest='legacy_roots')
            p.add_argument('--cost-root', action='append', dest='cost_roots')
            p.add_argument('--apply', action='store_true')
            p.add_argument('--watch', action='store_true')
            p.add_argument('--state-dir')
            p.add_argument('--poll-seconds', type=float, default=120)
            p.add_argument('--max-polls', type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == 'smoke':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_*.py',
                                                   top_level_dir=str(plan.ROOT))
        return not unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite).wasSuccessful()
    cfg = plan.default_config(args.server)
    if args.command == 'catalog':
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return 0
    root = Path(args.output).resolve() / plan.CAMPAIGN_ID / args.server / cfg['dataset']
    if args.command in ('publish', 'publish-auto'):
        from .publication import publish, start_publisher
        root = Path(args.root).resolve() if args.root else root
        if args.command == 'publish-auto':
            if not args.apply:
                parser.error('publish-auto requires explicit --apply (normal launcher supplies it)')
            # Host-side daemon, so legacy writer ownership checks refer to the
            # actual host repository and PID namespace even for Docker training.
            expires = time.monotonic() + 108000
            while not (root / 'preflight.json').exists():
                state = root / 'campaign_state.json'
                if state.exists() and json.loads(state.read_text()).get('status') in ('FAILED', 'STOPPED_BUDGET'):
                    raise RuntimeError('Training preflight stopped; no publisher was replaced')
                if time.monotonic() >= expires:
                    raise TimeoutError('No completed donor30 preflight; no publisher was replaced')
                time.sleep(10)
            while True:
                try:
                    result = start_publisher(root, args.server, credentials=args.credentials,
                        python_executable=sys.executable, legacy_roots=args.legacy_roots, cost_roots=args.cost_roots)
                    break
                except Exception as exc:
                    print(json.dumps({'publisher_bootstrap_error': repr(exc),
                        'training_untouched': True, 'retry_seconds': 60}), flush=True)
                    if time.monotonic() >= expires:
                        raise
                    time.sleep(60)
            print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
            return 0
        return publish(root, args.server, credentials=args.credentials, apply=args.apply, watch=args.watch,
            state_dir=args.state_dir, legacy_roots=args.legacy_roots, cost_roots=args.cost_roots,
            poll_seconds=args.poll_seconds, max_polls=args.max_polls)
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu(args.device)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    from .runtime import Campaign
    campaign = Campaign(cfg, output=args.output, device=args.device)
    state = campaign.run(catalog_path=Path(args.sources))
    print(json.dumps({'status': state['status'], 'root': str(campaign.root),
                      'elapsed_seconds': state['elapsed_seconds']}, ensure_ascii=False), flush=True)
    return 1 if state['status'] == 'FAILED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
