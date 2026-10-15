"""Real selected Student checkpoint and RR/FR input, CPU only; no production writes."""
import json, pathlib, sys, time
import numpy as np
import torch, yaml
ROOT=pathlib.Path('/home/knuvi/Desktop/song/PAN-Crafter'); OUT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT)); torch.set_num_threads(2)
from main import import_class
from tools.eval_fr_paperset import build
from tools import noa_eval as NE, s1_aligner_analysis as SA, qrecon24_select as SELECT
from kdv.eval_modes import CallCounter, forward_mode, state_hash, MODES
from unittest.mock import patch
run='PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S1234_FRESH50_v1'
wd=ROOT/'work_dir'/run; cfg=yaml.safe_load((wd/'meta/config.yaml').read_text())
m,_,how=build(cfg,str(wd),'best_hqnr');m.eval();h0=state_hash(m)
result=dict(run=run,checkpoint_sha256=NE.sha256_file(wd/'best_hqnr/model.safetensors'),forward=how,device='cpu',splits={})
for split in ['rr','fr']:
    ds=import_class(cfg['feeder'])(**cfg['test_reduced_feeder_args' if split=='rr' else 'test_full_feeder_args'])
    item=ds[0]; lms,ms,lpan,pan=item[1:] if split=='rr' else item
    pan,ms,lpan=[t[None] for t in (pan,ms,lpan)]; ys={}; rr={}; t0=time.time()
    for mode in MODES:
        inputs=[];views=[];grid=[]
        bh=m.backbone.register_forward_pre_hook(lambda mod,a:inputs.append([list(t.shape) for t in a]))
        ah=m.aligner.register_forward_pre_hook(lambda mod,a:views.append([list(t.shape) for t in a]))
        orig=torch.nn.functional.grid_sample
        def gs(*a,**k):grid.append(1);return orig(*a,**k)
        with CallCounter(m) as cc, patch('torch.nn.functional.grid_sample',gs),torch.no_grad():
            y,ex=forward_mode(m,pan,ms,lpan,mode,return_extra=True)
        bh.remove();ah.remove();ys[mode]=y
        rr[mode]=dict(calls=cc.as_dict(),grid_sample_calls=len(grid),backbone_inputs=inputs,aligner_input_shapes=sorted({str(v) for v in views}),extra=ex,sampler_restored=m.sampler)
    with torch.no_grad():
        base=torch.nn.functional.interpolate(ms,scale_factor=4,mode='bicubic')
        manual=base+m.backbone(pan,lpan,ms,torch.ones(1))
        offsets=[]
        for y in range(0,pan.shape[-2],64):
            for x in range(0,pan.shape[-1],64):
                offsets.append(m.predict_delta(pan[...,y:y+64,x:x+64],base[...,y:y+64,x:x+64])[0].numpy())
    median=np.median(np.array(offsets),axis=0)
    rr['checks']=dict(noa_manual_max_abs=float((manual-ys['A_BYPASS_RAW']).abs().max()),consensus_numpy_max_abs=float(np.max(np.abs(median-np.array(rr['A_CROP64_MED']['extra']['consensus_delta'])))),zero_noa_max_abs=float((ys['A_ZERO_WARP']-ys['A_BYPASS_RAW']).abs().max()),on_noa_max_abs=float((ys['A_ON']-ys['A_BYPASS_RAW']).abs().max()),seconds=time.time()-t0,n_scenes_available=len(ds))
    result['splits'][split]=rr; print(split,rr['checks'],flush=True)
result['state_unchanged']=h0==state_hash(m);result['no_grads']=all(p.grad is None for p in m.parameters())
# Analytic constant aligner: c_j == c_0, so q_size == average AXIS16 L1 / 2 == .46875.
class Constant:
    def predict_delta(self,p,m):return torch.zeros((1,2))
for split,n in [('rr',256),('fr',512)]:
    rows=SA.analyse_scene(Constant(),torch.zeros(1,1,n,n),torch.zeros(1,8,n//4,n//4),split,'cpu')
    result['probe_'+split]=dict(n_rows=len(rows),sizes={str(s):sum(r['size']==s for r in rows) for s in sorted({r['size'] for r in rows})},q_size_values=sorted({r['q_size'] for r in rows}),keys=list(rows[0]))
rows=[]
for scene,dy in [(0,0.),(1,100.)]:
    rows.extend(dict(scene=scene,size=64,q_size=1.,dy=dy,dx=0) for _ in range(2))
    rows.append(dict(scene=scene,size=256,q_size=1.,dy=dy,dx=0))
result['scene_aggregation_counterexample']=dict(expected_within_scene_crop_dy_std=0,reported=SA.summarise(rows,'rr')['by_size']['64']['dy_std'])
candidates=[dict(step=1010,hqnr=.959,ergas=2.039,scc=.988),dict(step=2020,hqnr=.960,ergas=2.070,scc=.989)]
result['selector_counterexample']=dict(selector=SELECT.SELECTOR,order=SELECT.ORDER,expected_step=1010,selected_step=SELECT.rank(candidates)[0]['step'])
(OUT/'math_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print('state',result['state_unchanged'],'grads absent',result['no_grads'],flush=True)
