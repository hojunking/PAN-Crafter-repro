"""Comparable batch1 FP32 cost: native MS inference, 3 scenes, 20+100 trials."""
import os
import time
import numpy as np
import torch
from torch.nn import functional as F
from tools.additional_models_flops import count_forward


def prepared(model,pan,ms,mode):
    base=F.interpolate(ms,size=pan.shape[-2:],mode='bicubic',align_corners=False)
    return base+model.residual(pan,base,mode)


def idle():
    from pcrepro.handoff import gpu_processes
    pids=gpu_processes()
    if pids is None or any(pid!=os.getpid() for pid in pids):
        raise RuntimeError('Concurrent or unknown CUDA computation invalidates cost measurement')


def stats(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),
                p05=float(np.percentile(a,5)),p95=float(np.percentile(a,95)),samples=len(values))


@torch.inference_mode()
def profile_model(model,datasets,device='cuda',stop_check=None):
    if torch.device(device).type!='cuda':raise ValueError('Published latency requires actual CUDA')
    if (model.hidden_size,tuple(model.depth),model.num_bands)!=(128,(2,2,4),4):
        raise ValueError('Finite-study cost requires the exact C4/C128/D224 model')
    was=model.training;model.eval();rows=[]
    def check():
        if stop_check and stop_check():raise InterruptedError('Safe cost pause')
        idle()
    try:
        for split,size in [('rr',256),('fr',512)]:
            data=datasets[split]
            if len(data)!=20:raise ValueError('RR20/FR20 cost scene population required')
            wall=[];events=[];peaks=[];per_scene=[];flops=None
            for i in [0,7,14]:
                check();batch=data[i]
                pan=batch['pan'][None].to(device);ms=batch['ms'][None].to(device)
                if pan.shape!=(1,1,size,size) or ms.shape!=(1,4,size//4,size//4):raise ValueError('Cost shape mismatch')
                mode=torch.ones(1,device=device,dtype=torch.long)
                fn=lambda:prepared(model,pan,ms,mode)
                expected=model(pan,ms);actual=fn()
                if not torch.equal(expected,actual):raise ValueError('Prepared/native model parity failed')
                del expected,actual
                if flops is None:flops=count_forward(fn)
                for _ in range(20):fn()
                torch.cuda.synchronize();check()
                scene_wall=[];scene_event=[]
                for j in range(100):
                    if j%20==0:check()
                    start=time.perf_counter();fn();torch.cuda.synchronize()
                    scene_wall.append((time.perf_counter()-start)*1000)
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                for j in range(100):
                    if j%20==0:check()
                    a.record();fn();b.record();b.synchronize();scene_event.append(a.elapsed_time(b))
                torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                y=fn();torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();del y
                peaks.append(peak);wall.extend(scene_wall);events.extend(scene_event)
                per_scene.append(dict(index=i,sync_wall_ms=stats(scene_wall),cuda_event_ms=stats(scene_event),
                    wall_samples_ms=scene_wall,event_samples_ms=scene_event,
                    baseline_allocated_bytes=baseline,peak_allocated_bytes=peak,native_parity_bit_equal=True))
                del batch,pan,ms,mode
                check()
            rows.append(dict(split=split.upper(),input_pan=[1,1,size,size],input_ms=[1,4,size//4,size//4],
                batch=1,precision='FP32; TF32 off',warmup=20,repeats=100,scenes=[0,7,14],
                infer_ms=stats(wall)['median'],sync_wall_ms=stats(wall),cuda_event_ms=stats(events),
                peak_memory_bytes=max(peaks),peak_allocated_MiB=max(peaks)/2**20,
                flops=flops,flops_g=flops['gflops'],per_scene=per_scene,
                gpu_name=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                scope='MS-mode model including internal bicubic MS resize and final residual add',
                exclusions=['external DN normalization','IO/H2D','finite/shape checks outside prepared model',
                            'metrics','training PAN-mode residual','warmup','profiler'],
                gpu_condition='no other CUDA compute PIDs; desktop graphics may remain active'))
        return dict(params_total=sum(p.numel() for p in model.parameters()),
            params_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
            params_m=sum(p.numel() for p in model.parameters())/1e6,profiles=rows,
            protocol='PCPAPER_ADDITIONAL_FP32_BATCH1_3SCENES_20WARMUP_100REPEATS_v1')
    finally:model.train(was)
