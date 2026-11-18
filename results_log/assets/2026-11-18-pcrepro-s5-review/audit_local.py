"""Read-only local data/model/metric audit. Does not train or access CUDA."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import json
import time
import numpy as np
import torch
from pcrepro.finite import manifest_for
from pcrepro.common import atomic_json,source_identity,apply_runtime_policy
from pcrepro.data import NativeDataset,CANONICAL_BANDS
from pcrepro.model import build_model,parameter_counts
from pcrepro.training import mars_loss
from pcrepro.finite_cost import prepared
from pcrepro.evaluation import rr_scene_metrics,fr_scene_metrics
from tools.metrics.eval_fr import load_dlpan
from tools.additional_models_flops import count_forward
from ta2.evaluation import rr_scene,fr_scene

torch.set_num_threads(1);apply_runtime_policy(ROOT)
out=Path(__file__).parent;result=dict(cuda_used=False,training_started=False,datasets={})
wald=load_dlpan(str(ROOT.parent/'DLPan-Toolbox'))
for sensor in ('QB','GF2'):
    started=time.monotonic();print(sensor+' full data verification started',flush=True)
    manifest=manifest_for(sensor,ROOT);atomic_json(out/(sensor+'_verified_manifest.json'),manifest)
    data={s:NativeDataset(manifest,s,ROOT) for s in ('train','rr','fr')}
    print(sensor+' hashes, finite tensors and FR MAT20 identity verified',flush=True)
    model=build_model(4,seed=2025,max_pixel=manifest['max_dn']);params=parameter_counts(model)
    assert params['total']==params['trainable']==7129476
    sample=data['train'][0]
    values=mars_loss(model,*(sample[k][None] for k in ('pan','ms','gt','pan_dn')))
    values['loss'].backward()
    assert torch.isfinite(values['loss']) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    loss=float(values['loss']);del values;model.zero_grad(set_to_none=True);model.eval()
    with torch.no_grad():model.output[-1].weight.normal_(0,.001)
    pan=sample['pan'][None];ms=sample['ms'][None];mode=torch.ones(1,dtype=torch.long)
    with torch.inference_mode():assert torch.equal(model(pan,ms),prepared(model,pan,ms,mode))
    arithmetic=count_forward(lambda:prepared(model,pan,ms,mode))
    atomic_json(out/(sensor+'_native64_operator_audit.json'),arithmetic)
    rr_raw=data['rr'].raw(0)
    pred=torch.nn.functional.interpolate(torch.from_numpy(rr_raw['ms'].copy())[None].float(),size=(256,256),mode='bicubic',align_corners=False)[0].numpy()
    pred=np.clip(pred,0,manifest['max_dn'])
    a=rr_scene_metrics(pred,rr_raw['gt'],sensor);b=rr_scene(pred,rr_raw['gt'],sensor)
    rr_delta={k:abs(float(a[k])-float(v)) for k,v in b.items()}
    raw=data['fr'].raw(0);pred=np.clip(raw['lms'],0,manifest['max_dn'])
    a=fr_scene_metrics(pred,raw,sensor,CANONICAL_BANDS[sensor],wald)
    b=fr_scene(pred,raw['pan'],raw['lms'],raw['ms'],sensor,wald)
    fr_delta={k:abs(float(a[k])-float(b[k])) for k in ('hqnr','d_s','d_lambda','jqm')}
    assert max(rr_delta.values())<1e-10 and max(fr_delta.values())<1e-10,(rr_delta,fr_delta)
    result['datasets'][sensor]=dict(counts={s:v['count'] for s,v in manifest['splits'].items()},
        paths={s:v['path'] for s,v in manifest['splits'].items()},parameters=params,
        cpu_native64_mars_pairs=1,cpu_native64_mars_batch=2,optimizer_updates=0,
        finite_forward_backward=True,smoke_loss=loss,prepared_native_parity_bit_equal=True,
        arithmetic_native64_gflops=arithmetic['gflops'],unsupported_float_operators=arithmetic['unsupported_operator_calls'],
        metric_parity_scope='scene0; identical bounded DN prediction; reference ta2.evaluation',
        rr_absolute_differences=rr_delta,fr_absolute_differences=fr_delta,seconds=time.monotonic()-started)
    for d in data.values():d.close()
    del model,data
    atomic_json(out/'local_audit.json',result);print(sensor+' CPU model and metric parity PASS',flush=True)
result['source_identity']=source_identity(ROOT);result['complete']=True
atomic_json(out/'local_audit.json',result)
print(json.dumps({s:dict(counts=v['counts'],parameters=v['parameters'],max_metric_difference=max(*v['rr_absolute_differences'].values(),*v['fr_absolute_differences'].values())) for s,v in result['datasets'].items()},indent=2))
