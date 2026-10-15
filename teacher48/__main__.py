"""python -m teacher48 --help; no action occurs merely by importing the package."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ta2.evaluation import atomic_json, digest, evaluator_identity
from .plan import (CAMPAIGN_ID, PARENT_CAMPAIGN_ID, METHOD_CONTRACT, LANES, LOSS_CASES, LOSS_ORDER, OPTIMIZER_CASES,
                   OPTIMIZER_ORDER, HORIZONS, evaluation_steps, resolve_case)
from .reporting import LocalOutbox, GSpreadTransport
from .runtime import Campaign, append_jsonl


def catalog(server):
    lane=LANES[server]
    return {'campaign_id':CAMPAIGN_ID,'parent_campaign_id':PARENT_CAMPAIGN_ID,
        **METHOD_CONTRACT,'server':server,**lane,
        'loss_cases':[{'case':f,'lambda_eps_RR':LOSS_CASES[f][0],'lambda_eps_FR':LOSS_CASES[f][0],
                       'lambda_abs':LOSS_CASES[f][1],'seed':lane['seeds'][0],'schedule':'H50',
                       'fresh':True,'priority':i+1} for i,f in enumerate(LOSS_ORDER[server])],
        'optimizer_menu':[{'case':t,'lr_A_multiplier':m[0],'lr_U_multiplier':m[1],
                          'weight_decay_multiplier':m[2],'anchor':'top_at_most2_F_loss_recipes',
                          'relative_to':'pinned_baseline_not_previous_T', 'fresh':True}
                         for t,m in OPTIMIZER_CASES.items()],
        'optimizer_priority':OPTIMIZER_ORDER[server],
        'schedules':{k:{'horizon':v,'evaluation_steps':evaluation_steps(k),'fresh':True} for k,v in HORIZONS.items()},
        'spreadsheet_tab':lane['dataset']+'-ablations','local_first':True,
        'retention':'STRICT_TARGET_ONLY','test_aware':True}


def publish(args):
    dataset=LANES[args.server]['dataset']
    root=Path(args.output)/CAMPAIGN_ID/args.server/dataset
    outbox=LocalOutbox(root)
    transport=None
    while True:
        try:
            if outbox.pending():
                if transport is None:
                    transport=GSpreadTransport.from_credentials(args.credentials,server=args.server,dataset=dataset)
                result=outbox.publish_once(transport,server=args.server,dataset=dataset,limit=32)
                print(json.dumps(result,ensure_ascii=False),flush=True)
            closed=(root/'publisher_close.json').exists() and json.loads((root/'publisher_close.json').read_text()).get('training_process_finished') is True
            if not args.watch or (closed and not outbox.pending()):
                return
        except Exception as exc:
            append_jsonl(root/'publisher_errors.jsonl',{'error':f'{type(exc).__name__}: {exc}','unix_time':time.time()})
            print(f'[Teacher48 publisher] pending retained: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
            transport=None
            if not args.watch:
                raise
        time.sleep(30)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Stage1 Teacher48: independent s1WV3 / s2QB / s3GF2, fixed64 ablations')
    sub=parser.add_subparsers(dest='command',required=True)
    inspect=sub.add_parser('inspect-assets',help='Read-only discovery of prior configs and native-source metadata')
    inspect.add_argument('--output',help='Optional JSON report file (not a training source manifest)')
    sub.add_parser('evaluator-id',help='Print installed evaluator identity and required baseline revision SHA256')
    cat=sub.add_parser('catalog',help='Write the16F/9T/3H recipe catalog, never start training')
    cat.add_argument('--server',choices=LANES,required=True)
    cat.add_argument('--output',required=True)
    for command in ('admit','run'):
        p=sub.add_parser(command,help='Full source/recipe admission'+(' and fresh independent48h campaign' if command=='run' else ' only; optional diagnostic'))
        p.add_argument('--server',choices=LANES,required=True)
        p.add_argument('--baseline',required=True)
        p.add_argument('--sources',required=True)
        p.add_argument('--output',default='work_dir/teacher48')
        p.add_argument('--device',default='cuda')
        if command=='run':
            p.add_argument('--credentials',help='Service-account file; omitted uses repository gspread/account.json')
            p.add_argument('--resume-campaign',action='store_true',help='Keep original48hclock; failed runs restart fresh with a new attempt')
    p=sub.add_parser('publish',help='Replay only this campaign durable outbox; never backfill old results')
    p.add_argument('--server',choices=LANES,required=True)
    p.add_argument('--output',default='work_dir/teacher48')
    p.add_argument('--credentials')
    p.add_argument('--watch',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='inspect-assets':
        from .asset_audit import discover_existing_assets
        report=discover_existing_assets(Path(__file__).resolve().parents[1])
        if args.output:
            atomic_json(args.output,report)
        print(json.dumps(report,ensure_ascii=False,indent=2)); return
    if args.command=='evaluator-id':
        obj=evaluator_identity()
        print(json.dumps({'evaluator_revision':digest(obj),'identity':obj},ensure_ascii=False,indent=2)); return
    if args.command=='catalog':
        obj=catalog(args.server)
        atomic_json(Path(args.output)/(args.server+'_catalog.json'),obj,immutable=True)
        print(json.dumps(obj,ensure_ascii=False,indent=2)); return
    if args.command=='publish':
        publish(args); return
    baseline=json.loads(Path(args.baseline).read_text())
    campaign=Campaign(args.output,args.server,baseline,device=args.device)
    if args.command=='admit':
        print(json.dumps(campaign.admission(args.sources),ensure_ascii=False,indent=2)); return
    # Reporting is a separate process: network time/locks never enter a train update.
    cmd=[sys.executable,'-m','teacher48','publish','--server',args.server,'--output',str(Path(args.output).resolve()),'--watch']
    if args.credentials:
        cmd+=['--credentials',str(Path(args.credentials).resolve())]
    started=[]
    def on_locked():
        atomic_json(campaign.root/'publisher_close.json',{'training_process_finished':False})
        with (campaign.root/'publisher.log').open('a') as log:
            process=subprocess.Popen(cmd,cwd=Path(__file__).resolve().parents[1],stdout=log,stderr=log,start_new_session=True)
        started.append(process.pid)
        atomic_json(campaign.root/f'publisher_process_{process.pid}.json',{'pid':process.pid,'command':cmd})
    try:
        result=campaign.launch(args.sources,resume_campaign=args.resume_campaign,on_locked=on_locked)
        print(json.dumps({'status':result['status'],'local_results':str(campaign.root),
                          'pending_uploads':len(campaign.outbox.pending())},ensure_ascii=False,indent=2))
    except BaseException as exc:
        if started:
            campaign.emit({},'CAMPAIGN_INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'CAMPAIGN_FAILED',
                          extra={'error':f'{type(exc).__name__}: {exc}','exact_resume_of_failed_weights':False})
            campaign.summarize()
        raise
    finally:
        # A failing campaign still has failure records to publish. The worker
        # exits only after its outbox is empty; auth failures stay retryable.
        if started:
            atomic_json(campaign.root/'publisher_close.json',{'training_process_finished':True})


if __name__=='__main__':
    main()
