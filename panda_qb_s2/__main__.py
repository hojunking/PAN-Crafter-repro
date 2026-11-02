"""QB_S2: native checks, profiling, training and existing Sheet publication."""
import argparse
import json
import sys
from pathlib import Path

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('catalog','check','check-sheet','cost','run','publish'))
    parser.add_argument('--server',choices=('s2',),required=True)
    parser.add_argument('--csv',action='store_true')
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args(argv)
    from . import plan,registry as G
    root=plan.ROOT/G.OUTPUT/G.CAMPAIGN_ID/args.server/'QB'
    if args.command=='check-sheet':
        from sheet_publish_v2.transport import GoogleSheets
        from .best_publication import project
        from .best_publication.transport import prepare
        root.mkdir(parents=True,exist_ok=True)
        value=project(root,args.server)
        adapter=GoogleSheets('gspread/account.json')
        snapshot=adapter.snapshot(value['destination'])
        prepare(snapshot,value,args.server)
        from .best_publication.header import reconcile_header
        reconcile_header(adapter,value['destination'],root/'sheet_publish_panda_qb_s2',args.server,apply=False)
        print(json.dumps({'status':'SHEET_READ_ONLY_CHECK_OK','destination':value['destination']}));return 0
    if args.command=='catalog':
        print(plan.catalog_csv(args.server) if args.csv else json.dumps(plan.default_config(args.server),indent=2));return 0
    if args.command=='publish':
        from .best_publication.__main__ import publish
        return publish(root,args.server,apply=args.apply,watch=args.watch)
    if args.command in ('run','cost') and (plan.ROOT/'gspread/server.txt').read_text().strip()!=args.server:
        raise ValueError('Requested server differs from local marker')
    if args.command=='check':
        from .release import verify
        verify(check_environment=True)
        from panda_stage2_condq.sources import RegisteredRRTrain
        from ta2.evaluation import evaluator_identity,file_sha
        from .runtime import code_pin
        from .reporting import _config
        cfg=plan.default_config(args.server)
        catalog=plan.ROOT/'ablr2/sensor_sources.json'
        if file_sha(catalog)!=cfg['baseline_resolved']['source_registry_sha256']:raise ValueError('Data catalog changed')
        reg=RegisteredRRTrain.from_catalog(catalog,root=plan.ROOT,sensor='QB',lms_overrides=plan.lms_overrides('QB'))
        import h5py
        counts={}
        for split in ('train','val','rr','fr'):
            reg.sources.raw(split,[0])
            with h5py.File(reg.sources._verify(split),'r') as handle:
                counts[split]=len(handle['pan'])
        if counts!={'train':17139,'val':1905,'rr':20,'fr':20}:
            raise ValueError(f'Wrong QB split populations: {counts}')
        if len(reg.sample_ids)!=17139:raise ValueError('Wrong QB training population')
        for c,r in G.RUNS[args.server]:_config(plan.resolve_case(args.server,c,r))
        code_pin();evaluator_identity()
        print(json.dumps(dict(status='READ_ONLY_CHECK_OK',server=args.server,dataset='QB',train_count=len(reg.sample_ids),split_counts=counts,
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
