"""Train independent local s3 QB Stage1, select by validation, then eight Students."""
import argparse
import json
import sys
from . import registry as G


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('catalog', 'check', 'cost', 'run', 'publish', 'verify-donor'))
    parser.add_argument('--server', choices=('s3',), default='s3')
    parser.add_argument('--csv', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    from .adapter import install
    plan = install()
    root = plan.ROOT / G.OUTPUT / G.CAMPAIGN_ID / 's3/QB'
    if args.command == 'catalog':
        print(plan.catalog_csv('s3') if args.csv else json.dumps(plan.default_config('s3'), indent=2))
        return 0
    if args.command == 'publish':
        from panda_s1loss9.best_publication.__main__ import publish
        return publish(root, 's3', apply=args.apply, watch=args.watch)
    if args.command == 'verify-donor':
        from .selected import verify
        print(json.dumps(verify(plan.ROOT)))
        return 0
    from .prerequisites import evidence
    evidence(plan.ROOT)
    if args.command == 'cost':
        from panda_s1loss9.cost.__main__ import main as cost_main
        sys.argv = ['cost', '--server', 's3', '--root', str(root), '--seconds', '1200']
        return cost_main()
    if args.command == 'check':
        from panda_stage2_condq.sources import RegisteredRRTrain
        from ta2.evaluation import file_sha, evaluator_identity
        from panda_s1loss9.runtime import code_pin
        catalog = plan.ROOT / 'ablr2/sensor_sources.json'
        cfg = plan.default_config('s3')
        if file_sha(catalog) != cfg['baseline_resolved']['source_registry_sha256']:
            raise ValueError('Registered QB catalog changed')
        registered = RegisteredRRTrain.from_catalog(catalog, root=plan.ROOT, sensor='QB',
                                                     lms_overrides=plan.lms_overrides('QB'))
        for split in ('val', 'rr', 'fr'): registered.sources.raw(split, [0])
        import h5py
        with h5py.File(registered.sources._verify('val')) as handle: val_count = len(handle['pan'])
        if len(registered.sample_ids) != 17139 or val_count != 1905:
            raise ValueError('QB train/validation populations differ')
        code_pin(); evaluator_identity()
        print(json.dumps({'status': 'READ_ONLY_CHECK_OK', 'server': 's3', 'dataset': 'QB',
                          'stage1_runs': 1, 'stage2_runs': 8, 'stage1_seed': G.STAGE1_SEED}))
        return 0
    if (plan.ROOT / 'gspread/server.txt').read_text().strip() != 's3':
        raise ValueError('Only s3 may train this independent Stage1 campaign')
    from directshift48.__main__ import require_idle_gpu
    from panda_s1loss9.runtime import Campaign
    require_idle_gpu('cuda')
    campaign = Campaign('s3', output=plan.ROOT / G.OUTPUT)
    state = campaign.run()
    print(json.dumps({'root': str(campaign.root), 'status': state['status']}))
    return int(state['status'] not in ('FINISHED_REGISTERED_QUEUE', 'FINISHED_WITH_EXPLICIT_EARLY_STOPS'))


if __name__ == '__main__': raise SystemExit(main())
