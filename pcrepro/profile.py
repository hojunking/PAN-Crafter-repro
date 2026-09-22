"""Native MS-only timing; explicit partial operation-count convention."""
import math
import time
import torch

@torch.no_grad()
def profile_model(model,bands,device='cuda',sizes=(256,512),stop_check=None):
    dev=torch.device(device);was_training=model.training;model.eval();rows=[]
    try:
        for size in sizes:
            if stop_check and stop_check():raise InterruptedError('Safe profile pause')
            pan=torch.zeros(1,1,size,size,device=dev);ms=torch.zeros(1,bands,size//4,size//4,device=dev)
            macs=[0];handles=[]
            def hook(layer,inputs,out):
                if isinstance(layer,torch.nn.Linear):macs[0]+=out.numel()*layer.in_features
                elif isinstance(layer,torch.nn.ConvTranspose2d):macs[0]+=inputs[0].numel()*(layer.out_channels//layer.groups)*math.prod(layer.kernel_size)
                else:macs[0]+=out.numel()*(layer.in_channels//layer.groups)*math.prod(layer.kernel_size)
            try:
                for layer in model.modules():
                    if isinstance(layer,(torch.nn.Conv2d,torch.nn.ConvTranspose2d,torch.nn.Linear)):handles.append(layer.register_forward_hook(hook))
                model(pan,ms)
            finally:
                for h in handles:h.remove()
            # CM3A attention dot/value sums (two modalities); score softmax excluded.
            attention_macs=4*128*9*((size//2)**2*2+(size//4)**2)
            for _ in range(3):model(pan,ms)
            if dev.type=='cuda':torch.cuda.synchronize(dev);torch.cuda.reset_peak_memory_stats(dev)
            start=time.perf_counter()
            for _ in range(10):
                if stop_check and stop_check():raise InterruptedError('Safe profile pause')
                model(pan,ms)
            if dev.type=='cuda':torch.cuda.synchronize(dev)
            rows.append(dict(input_pan=[1,1,size,size],input_ms=[1,bands,size//4,size//4],batch=1,
                warmup=3,repeats=10,synchronized=dev.type=='cuda',infer_ms=(time.perf_counter()-start)*100,
                peak_memory_bytes=torch.cuda.max_memory_allocated(dev) if dev.type=='cuda' else None,
                conv_linear_macs=macs[0],attention_macs=attention_macs,
                macs_partial_g=(macs[0]+attention_macs)/1e9,flops_partial_g=2*(macs[0]+attention_macs)/1e9,
                count_complete=False,convention='FLOPs=2*MACs; Conv/ConvTranspose/Linear plus local QK/AV, excludes interpolation/norm/activation/softmax/bias/residual arithmetic',
                mode='MS_ONLY_NO_PAN_RESIDUAL',device=str(dev),gpu_name=torch.cuda.get_device_name(dev) if dev.type=='cuda' else 'CPU',
                precision='FP32',paper_gpu_comparable=False))
    finally:model.train(was_training)
    return dict(params_total=sum(p.numel() for p in model.parameters()),
        params_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        params_m=sum(p.numel() for p in model.parameters())/1e6,profiles=rows)
