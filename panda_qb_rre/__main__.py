"""QB_RRE: native checks, profiling, training and existing Sheet publication."""
import argparse
import json
import sys
from pathlib import Path

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('catalog','check','cost','run','publish','verify-donor'))
    parser.add_argument('--server',choices=('s1','s3'),required=True)
    parser.add_argument('--csv',action='store_true')
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args(argv)
    # Source dispatch must be active before importing the historical Stage1.
    # It does not change the new QB_RRE registry or any source file/tensor.
    if args.server=='s3':
        from panda_qb_deepen.adapter import install
        install()
    from . import plan,registry as G
    root=plan.ROOT/G.OUTPUT/G.CAMPAIGN_ID/args.server/'QB'
    if args.command=='catalog':
        print(plan.catalog_csv(args.server) if args.csv else json.dumps(plan.default_config(args.server),indent=2));return 0
    if args.command=='publish':
        from .best_publication.__main__ import publish
        return publish(root,args.server,apply=args.apply,watch=args.watch)
    if args.command=='verify-donor':
        from .external import verify
        print(json.dumps(verify(plan.ROOT,args.server)));return 0
    if (plan.ROOT/'gspread/server.txt').read_text().strip()!=args.server:
        raise ValueError('Requested server differs from local marker')
    if args.command=='check':
        from panda_stage2_condq.sources import RegisteredRRTrain
        from ta2.evaluation import evaluator_identity,file_sha
        from .runtime import code_pin
        from .reporting import _config
        cfg=plan.default_config(args.server)
        catalog=plan.ROOT/'ablr2/sensor_sources.json'
        if file_sha(catalog)!=cfg['baseline_resolved']['source_registry_sha256']:raise ValueError('Data catalog changed')
        reg=RegisteredRRTrain.from_catalog(catalog,root=plan.ROOT,sensor='QB',lms_overrides=plan.lms_overrides('QB'))
        for split in ('val','rr','fr'):reg.sources.raw(split,[0])
        if len(reg.sample_ids)!=17139:raise ValueError('Wrong QB training population')
        for c,r in G.RUNS[args.server]:_config(plan.resolve_case(args.server,c,r))
        code_pin();evaluator_identity()
        print(json.dumps(dict(status='READ_ONLY_CHECK_OK',server=args.server,dataset='QB',train_count=len(reg.sample_ids),
            stage1_runs=cfg['stage1_runs'],stage2_runs=cfg['stage2_runs'],joint_target=G.TARGET)));return 0
    if args.command=='cost':
        from .cost.__main__ import main as cost_main
        sys.argv=['cost','--server',args.server,'--root',str(root),'--seconds','1200']
        return cost_main()
    from directshift48.__main__ import require_idle_gpu
    from .runtime import Campaign
    from .runner import TERMINAL
    require_idle_gpu('cuda')
    camp=Campaign(args.server,output=str(plan.ROOT/G.OUTPUT))
    result=camp.run()
    print(json.dumps(dict(root=str(root),status=result['status'])))
    return int(result['status'] not in TERMINAL)

if __name__=='__main__':raise SystemExit(main())
