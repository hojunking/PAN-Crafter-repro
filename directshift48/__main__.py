"""Independent s1/WV3 or s3/GF2 controller, never a remote launch command."""
import argparse
import json
from pathlib import Path
import subprocess

from . import plan

ROOT = Path(__file__).resolve().parents[1]


def require_idle_gpu(device):
    """Fail closed before importing/initializing CUDA; never stop a workload."""
    if not str(device).startswith('cuda'):
        return
    try:
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader'], text=True, stderr=subprocess.STDOUT, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError('Cannot verify idle GPU; no training launched') from exc
    if processes.strip():
        raise RuntimeError('Active GPU compute workload; no process was stopped and no new training launched:\n'+processes)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prior-free DirectShift / Scharr / R3, independent 48h lanes')
    sub = parser.add_subparsers(dest='command', required=True)
    smoke = sub.add_parser('smoke', help='CPU-only synthetic contracts; no data, Sheets or training launch')
    smoke.add_argument('--verbose', action='store_true')
    for name in ('run', 'admit', 'catalog', 'publish', 'retry-artifacts'):
        command = sub.add_parser(name)
        command.add_argument('--server', required=True, choices=plan.LANES)
        command.add_argument('--output', default='work_dir/directshift48')
        if name in ('run', 'admit', 'retry-artifacts'):
            command.add_argument('--baseline')
            command.add_argument('--sources', default='ablr2/sensor_sources.json')
            command.add_argument('--device', default='cuda')
        if name == 'publish':
            command.add_argument('--credentials', default='gspread/account.json')
            command.add_argument('--history-root', action='append', dest='history_roots')
            command.add_argument('--apply', action='store_true', help='Explicitly permit existing-tab projection writes')
            command.add_argument('--watch', action='store_true')
            command.add_argument('--state-dir')
            command.add_argument('--poll-seconds', type=float, default=120)
    args = parser.parse_args(argv)
    if args.command == 'smoke':
        from .smoke import run_smoke
        return run_smoke(verbose=args.verbose)
    if args.command == 'catalog':
        print(json.dumps(plan.catalog(args.server), ensure_ascii=False, indent=2))
        return 0
    if args.command == 'publish':
        from .publication import publish
        root = Path(args.output).resolve()/plan.CAMPAIGN_ID/args.server/plan.LANES[args.server]['dataset']
        return publish(root, args.server, credentials=args.credentials, apply=args.apply,
            watch=args.watch, state_dir=args.state_dir, history_roots=args.history_roots,
            poll_seconds=args.poll_seconds)
    if args.command == 'run':
        require_idle_gpu(args.device)
    from .runtime import Campaign, build_trainer
    baseline = Path(args.baseline) if args.baseline else ROOT/'config/directshift48'/f'{args.server}.baseline.json'
    campaign = Campaign(args.output, args.server, json.loads(baseline.read_text()), device=args.device)
    if args.command == 'admit':
        print(json.dumps(campaign.admission(args.sources), ensure_ascii=False, indent=2))
        return 0
    if args.command == 'retry-artifacts':
        import torch
        from .evaluation import retry_qualified
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        campaign.admission(args.sources)
        for path in sorted(campaign.root.glob('runs/**/artifact_manifest.json')):
            manifest = json.loads(path.read_text())
            if manifest.get('status') == 'QUAL_PENDING':
                result = retry_qualified(path.parent, campaign.sources,
                    build_trainer=lambda c: build_trainer(c, campaign.sources, args.device), device=args.device)
                print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    campaign.execute(args.sources)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
