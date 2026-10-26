"""CPU diagnostic of corrected-minus-original PAN input magnitude, same ROI size."""
import json
from pathlib import Path
import sys
root=Path('/home/knuvi/Desktop/song/PAN-Crafter')
sys.path[:0]=['/tmp/panda_sensor5_v1_qb_audit',str(root)]
import h5py
import numpy as np
import torch
from panda_sensor5.aligner import BandPANAligner
from panda_sensor5.warp import warp_pan
torch.set_num_threads(4)
cp=next((root/'work_dir/panda_sensor5').glob('PANDA_S13_SENSOR5_STAGE12_20261026_v1/s1/QB/runs/*G4*/attempt_01/models/step_0038000.pt'))
state=torch.load(cp,map_location='cpu',weights_only=False)['trainer']['stage']
aligner=BandPANAligner().eval()
aligner.load_state_dict({k[8:]:v for k,v in state.items() if k.startswith('aligner.')})
entry=json.loads((root/'ablr2/sensor_sources.json').read_text())['sensors']['QB']
report={'purpose':'DIAGNOSTIC_ONLY_NO_TRAINING','checkpoint':str(cp.relative_to(root)),
        'population':'128_evenly_spaced_unaugmented_RR_train_patches; all20_native_RR_and_FR_scenes',
        'warp':'Student_lanczos4_reflection','measurement_roi':'center64_then_inner48_same_size_all_splits',
        'normalization':'model=2*DN/2047-1; relative=mean_abs_difference/PAN_population_std_same_ROI',
        'caveat':'Observed marginal input distribution only, not a causal test or full augmentation distribution',
        'splits':{}}
with torch.inference_mode():
    for split in ('train','rr','fr'):
        rows=[]
        with h5py.File(root/entry['splits'][split]['path'],'r') as f:
            original=h5py.File(root/'data/PanCollection/QB/train_qb.h5','r') if split=='train' else f
            indices=np.linspace(0,len(f['pan'])-1,128,dtype=int).tolist() if split=='train' else list(range(20))
            for i in indices:
                pan=torch.from_numpy(np.asarray(f['pan'][i],np.float32)).unsqueeze(0)*(2/2047)-1
                lms=torch.from_numpy(np.asarray(original['lms'][i],np.float32)).unsqueeze(0)*(2/2047)-1
                u=aligner(pan,lms)
                corrected=warp_pan(pan,u)
                h,w=pan.shape[-2:];y=(h-64)//2+8;x=(w-64)//2+8
                p=pan[...,y:y+48,x:x+48];diff=(corrected-pan)[...,y:y+48,x:x+48]
                l1=float(diff.abs().mean());std=float(p.std(unbiased=False))
                rows.append({'index':i,'u':u[0].tolist(),'u_norm':float(u.norm()),
                    'abs_difference_DN':l1*2047/2,'abs_difference_model_units':l1,
                    'pan_std_DN':std*2047/2,'abs_difference_over_pan_std':l1/std if std>1e-12 else None})
            if split=='train': original.close()
        result={'count':len(rows),'per_sample':rows}
        for name in ('u_norm','abs_difference_DN','abs_difference_model_units','pan_std_DN','abs_difference_over_pan_std'):
            a=np.asarray([r[name] for r in rows if r[name] is not None])
            result[name]={'mean':float(a.mean()),'median':float(np.median(a)),
                          'p10':float(np.quantile(a,.1)),'p90':float(np.quantile(a,.9)),
                          'min':float(a.min()),'max':float(a.max())}
        report['splits'][split]=result
        print(split,{k:v for k,v in result.items() if k!='per_sample'},flush=True)
(root/'results_log/assets/2026-10-26_qb-diff-distribution.json').write_text(json.dumps(report,indent=2)+'\n')
