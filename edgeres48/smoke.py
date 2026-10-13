"""Disposable real-data integration test; no model/image save or Sheets upload."""
import argparse
import json
from pathlib import Path
import time
import torch

from .evaluation import official_evaluation, model_state_hash
from .plan import resolve_case
from .reporting import atomic_json
from .runtime import Campaign, build_trainer, ROOT, probe_spec
from .stream import RRStream, set_learning_rates


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',choices=('s1','s2','s3'),required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--full-eval',action='store_true')
    parser.add_argument('--probes',action='store_true')
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if not 1<=args.steps<=10: raise ValueError('Disposable smoke allows 1..10 updates')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    baseline=json.loads((ROOT/'config/edgeres48'/f'{args.server}.baseline.json').read_text())
    output=Path(args.output)
    campaign=Campaign(output/'admission',args.server,baseline,device=args.device)
    campaign.admission(ROOT/'ablr2/sensor_sources.json')
    config=resolve_case(args.server,baseline=baseline)
    trainer=build_trainer(config,campaign.sources,args.device)
    stream=RRStream(campaign.sources,config,device=args.device)
    result={'disposable':True,'eligible_for_campaign_selection':False,'weights_saved':False,
        'server':args.server,'dataset':campaign.dataset,'steps':[],
        'initial_model_state_sha256':model_state_hash(trainer.stage)}
    for _ in range(args.steps):
        started=time.monotonic();set_learning_rates(trainer,config)
        log=trainer.step_accumulated(stream.next_update())
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        result['steps'].append({'wall_seconds':time.monotonic()-started,**log})
        print(f"smoke {args.server} update={trainer.updates} loss={log['loss']}",flush=True)
    if args.full_eval:
        record,_=official_evaluation(trainer,config,campaign.sources,device=args.device,
                                     evaluator_pin=campaign.evaluator_pin)
        record.update(disposable=True,eligible_for_campaign_selection=False)
        result['evaluation']=record
    else:
        trainer.stage.eval()
        with torch.no_grad():
            result['native_shapes']={}
            for split in ('rr','fr'):
                prediction=trainer.stage.native_prediction(campaign.sources.batch(split,[0],device=args.device))
                result['native_shapes'][split]=list(prediction['y'].shape)
    result['final_model_state_sha256']=model_state_hash(trainer.stage)
    if args.probes:
        from .diagnostics import run_probes, benchmark_inference
        result['diagnostics']=run_probes(trainer,campaign.sources,probe_spec(config,clean=True),
            device=args.device,evaluator_pin=campaign.evaluator_pin)
        result['cost']=benchmark_inference(trainer,campaign.sources,device=args.device,
            warmup=3,repeats=10,rows={'rr':0,'fr':0})
        if result['diagnostics']['status']!='COMPLETE':
            raise RuntimeError('Disposable diagnostics did not complete within the declared budget')
    if args.device.startswith('cuda'):
        result['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
    atomic_json(output/f'{args.server}_smoke.json',result)
    print(json.dumps({'status':'PASS','report':str(output/f'{args.server}_smoke.json')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
