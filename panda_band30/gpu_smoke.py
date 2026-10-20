"""Disposable real-data CUDA checks; never publishes or supplies a donor."""
from pathlib import Path
import tempfile
import time

import torch

from directshift48.reporting import atomic_json
from donor30.evaluation import model_state_hash
from panda_method.edge_data import EdgeDataSources
from panda_stage2_condq.training import configure_precision
from ta2.evaluation import evaluator_identity, file_sha
from .plan import ROOT, resolve_case
from .stage1 import build_stage1, Stage1Stream, advance_stage1, save_stage1, load_stage1


def main():
    configure_precision()
    if not torch.cuda.is_available(): raise RuntimeError('Actual CUDA validation required')
    sources=EdgeDataSources(ROOT/'ablr2/sensor_sources.json',root=ROOT,sensor='WV3')
    results=[]; destination=ROOT/'work_dir/panda_band30/preflight'
    destination.mkdir(parents=True,exist_ok=True)
    for case in ('G8_S1_U23','B8A_S1_U23','G8_S1_BIC'):
        cfg=resolve_case('s1',case)
        cfg['disposable_preflight']=True
        cfg['run_id']+='__DISPOSABLE_NOT_A_DONOR'
        trainer=build_stage1(cfg,sources,device='cuda')
        stream=Stage1Stream(sources,cfg,device='cuda')
        torch.cuda.reset_peak_memory_stats()
        tick=time.monotonic()
        for _ in range(3): advance_stage1(trainer,stream)
        pins={'code_pin':{'panda_band30/gpu_smoke.py':file_sha(__file__)},
              'evaluator_pin':evaluator_identity()}
        with tempfile.TemporaryDirectory(prefix='band30-gpu-',dir=destination) as tmp:
            path=Path(tmp)/'step.pt'
            saved=save_stage1(path,trainer,stream,**pins)
            advance_stage1(trainer,stream)
            expected=model_state_hash(trainer.stage)
            expected_tensors={k:v.detach().cpu().clone() for k,v in trainer.stage.state_dict().items()
                              if isinstance(v,torch.Tensor)}
            del trainer,stream
            torch.cuda.empty_cache()
            trainer,stream,_=load_stage1(path,file_sha(path),cfg,sources,device='cuda',**pins)
            advance_stage1(trainer,stream)
            bitwise=model_state_hash(trainer.stage)==expected
            max_error=max(float((v.detach().cpu().double()-expected_tensors[k].double()).abs().max())
                          for k,v in trainer.stage.state_dict().items() if isinstance(v,torch.Tensor))
            # CUDA grid_sample backward permits floating-order differences. The
            # CPU test checks exact equality; CUDA must report, not hide, them.
            if max_error>1e-6:raise RuntimeError(f'CUDA resume divergence {case}: {max_error}')
            del expected_tensors
        shapes={}
        for split in ('rr','fr'):
            batch=sources.batch(split,[0],device='cuda')
            if split=='fr' and batch.gt is not None:raise ValueError('FR GT forbidden')
            with torch.no_grad():out=trainer.stage.native_prediction(batch)
            if not bool(torch.isfinite(out['y']).all()):raise ValueError('Nonfinite native prediction')
            shapes[split]={'output':list(out['y'].shape),'shift':list(out['u'].shape)}
            del out,batch
        if case in ('G8_S1_U23','B8A_S1_U23'):
            from .evaluation import official_evaluate
            report=official_evaluate(trainer,sources,evaluator_pin=pins['evaluator_pin'],device='cuda')
            if report['qualification']['metrics_complete'] is not True:
                raise ValueError('Official all40 metrics incomplete')
            if report['qualification']['same_state_verified'] is not True:
                raise ValueError('Official same-state validation failed')
            atomic_json(destination/(case+'_disposable_evaluation.json'),report)
            from .diagnostics import residual_registration
            diagnostic=residual_registration(destination,cfg,report,sources)
            print({'case':case,'disposable_registration':diagnostic},flush=True)
        torch.cuda.synchronize()
        result={'case':case,'next_update_bitwise_resume':bitwise,
                'next_update_max_abs_error':max_error,'steps':trainer.updates,
                'shapes':shapes,'elapsed_seconds':time.monotonic()-tick,
                'peak_cuda_mib':torch.cuda.max_memory_allocated()/1024**2,
                'published':False,'eligible_donor':False}
        results.append(result);print(result,flush=True)
        del trainer,stream
        torch.cuda.empty_cache()
    atomic_json(destination/'gpu_smoke.json',{'status':'PASS','device':torch.cuda.get_device_name(),
                                             'disposable_not_experimental_evidence':True,'results':results})


if __name__=='__main__':main()
