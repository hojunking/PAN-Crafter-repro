"""One command per server; publisher is independent of the training process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .plan import LANES, CAMPAIGN_ID, catalog
from .reporting import LocalOutbox, GSpreadTransport, atomic_json, append_jsonl

ROOT = Path(__file__).resolve().parents[1]


def publish(args):
    dataset=LANES[args.server]['dataset']
    directory=Path(args.output).resolve()/CAMPAIGN_ID/args.server/dataset
    outbox=LocalOutbox(directory)
    transport=None
    while True:
        try:
            outbox.recover()
            if outbox.pending():
                if transport is None:
                    transport=GSpreadTransport.from_credentials(args.credentials,server=args.server,dataset=dataset)
                print(json.dumps(outbox.publish_once(transport,server=args.server,dataset=dataset,limit=32)),flush=True)
            marker=directory/'publisher_close.json'
            closed=marker.exists() and json.loads(marker.read_text()).get('training_process_finished') is True
            if not args.watch or (closed and not outbox.pending()): return
        except Exception as exc:
            append_jsonl(directory/'publisher_errors.jsonl',{'error':f'{type(exc).__name__}: {exc}','unix_time':time.time()})
            print(f'[EdgeRes48 publisher] pending retained: {exc}',file=sys.stderr,flush=True)
            transport=None
            if not args.watch: raise
        time.sleep(30)


def main(argv=None):
    parser=argparse.ArgumentParser(description='EdgeRes48H-v2: RR-only independent s1/WV3, s2/QB, s3/GF2')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('run','admit','publish','catalog','retry-artifacts'):
        p=sub.add_parser(name)
        p.add_argument('--server',choices=LANES,required=True)
        p.add_argument('--output',default='work_dir/edgeres48')
        if name in ('run','admit','retry-artifacts'):
            p.add_argument('--baseline',help='Omit for config/edgeres48/<server>.baseline.json')
            p.add_argument('--sources',default='ablr2/sensor_sources.json')
            p.add_argument('--device',default='cuda')
        if name=='run': p.add_argument('--decisions',help='Optional append-only evidence-backed conditional decision JSON')
        if name in ('run','publish'): p.add_argument('--credentials')
        if name=='publish': p.add_argument('--watch',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='catalog':
        print(json.dumps(catalog(args.server),ensure_ascii=False,indent=2));return
    if args.command=='publish': publish(args);return
    from .runtime import Campaign
    baseline=Path(args.baseline) if args.baseline else ROOT/'config/edgeres48'/f'{args.server}.baseline.json'
    campaign=Campaign(args.output,args.server,json.loads(baseline.read_text()),device=args.device)
    if args.command=='admit':
        print(json.dumps(campaign.admission(args.sources),ensure_ascii=False,indent=2));return
    if args.command=='retry-artifacts':
        import torch
        from .evaluation import retry_qualified
        from .runtime import build_trainer
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        campaign.admission(args.sources)
        for path in sorted(campaign.root.glob('runs/**/artifact_manifest.json')):
            manifest=json.loads(path.read_text())
            if manifest.get('status')=='QUAL_PENDING':
                result=retry_qualified(path.parent,campaign.sources,
                    lambda c:build_trainer(c,campaign.sources,args.device),device=args.device)
                print(json.dumps(result,ensure_ascii=False),flush=True)
        return
    def publisher_start():
        atomic_json(campaign.root/'publisher_close.json',{'training_process_finished':False})
        command=[sys.executable,'-m','edgeres48','publish','--server',args.server,
                 '--output',str(Path(args.output).resolve()),'--watch']
        if args.credentials: command.extend(['--credentials',str(Path(args.credentials).resolve())])
        with (campaign.root/'publisher.log').open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(campaign.root/'publisher_process.json',{'pid':child.pid,'command':command})
    campaign.execute(args.sources,on_locked=publisher_start,decisions_path=args.decisions)


if __name__=='__main__': main()
