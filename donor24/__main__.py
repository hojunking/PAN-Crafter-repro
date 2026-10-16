"""One explicit local launch; catalog/smoke never starts training or uploads."""
import argparse
import json
import os
from pathlib import Path
import time

from . import plan


def main(argv=None):
    parser = argparse.ArgumentParser(description='24h WV3 Stage 1 donor search; s1 and s3')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('smoke')
    sub.add_parser('mounts')
    for name in ('catalog', 'run', 'publish', 'publish-auto'):
        p = sub.add_parser(name)
        p.add_argument('--server', required=True, choices=plan.LANES)
        p.add_argument('--output', default='work_dir/donor24')
        if name == 'catalog':
            p.add_argument('--csv', action='store_true')
        if name == 'run':
            p.add_argument('--device', default='cuda')
            p.add_argument('--sources', default='ablr2/sensor_sources.json')
        if name in ('publish', 'publish-auto'):
            p.add_argument('--credentials', default='gspread/account.json')
            p.add_argument('--apply', action='store_true')
            p.add_argument('--watch', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'smoke':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import unittest
        tests = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_*.py',
                                                   top_level_dir=str(plan.ROOT))
        return not unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful()
    if args.command == 'mounts':
        from edgeres48.docker_mounts import targets
        # Resolve WV3 assets on BOTH servers; never legacy s3->GF2 mapping.
        for path in targets(plan.ROOT, 's1'):
            print(path)
        return 0
    cfg = plan.load_config(args.server)
    if args.command == 'catalog':
        print(plan.catalog_csv() if args.csv else json.dumps({
            'campaign': cfg, 'search': [plan.resolve_case(cfg, c) for c in cfg['search_order']],
            'confirmation': 'same local winning recipe / R2, reserved before search',
            'stage2': 'not launched; one shared donor requires cross-server handoff'}, indent=2))
        return 0
    root = Path(args.output).resolve()/plan.CAMPAIGN_ID/args.server/'WV3'
    if args.command in ('publish', 'publish-auto'):
        from .publication import publish
        auto = args.command == 'publish-auto'
        if auto and not args.apply:
            parser.error('publish-auto requires explicit --apply')
        if auto:
            until = time.monotonic()+86400
            while not (root/'preflight.json').exists():
                status = root/'campaign_state.json'
                if status.exists() and json.loads(status.read_text()).get('status') in ('FAILED','STOPPED_BUDGET'):
                    raise RuntimeError('Preflight stopped; old sheet publisher untouched')
                if time.monotonic() >= until:
                    raise TimeoutError('No WV3 preflight; no publisher handoff')
                time.sleep(10)
        while True:
            try:
                return publish(root, args.server, credentials=args.credentials, apply=args.apply,
                               watch=args.watch or auto, handoff=auto)
            except Exception as exc:
                if not auto or time.monotonic() >= until:
                    raise
                print(json.dumps({'publisher_bootstrap_error': repr(exc),
                    'training_untouched': True, 'retry_seconds': 30}), flush=True)
                time.sleep(30)
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu(args.device)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    from .runtime import Campaign
    campaign = Campaign(cfg, output=args.output, device=args.device)
    status = campaign.run(catalog_path=Path(args.sources))
    print(json.dumps({'root': str(campaign.root), 'status': status['status']}, ensure_ascii=False))
    return 1 if status['status'] == 'FAILED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
