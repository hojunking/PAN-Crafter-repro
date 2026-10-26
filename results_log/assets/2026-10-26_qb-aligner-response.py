"""Frozen v1 Aligner ±1px response diagnostic; no optimization or publication."""
import json
from pathlib import Path
import sys
import time
root = Path('/home/knuvi/Desktop/song/PAN-Crafter')
sys.path[:0] = ['/tmp/panda_sensor5_v1_qb_audit', str(root)]
import h5py
import numpy as np
import torch
from panda_sensor5.aligner import BandPANAligner
from pa.warp import warp_pan

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
p = next((root/'work_dir/panda_sensor5').glob('PANDA_S13_SENSOR5_STAGE12_20261026_v1/s1/QB/runs/*G4*/attempt_01/models/step_0038000.pt'))
checkpoint = torch.load(p, map_location='cpu', weights_only=False)
sd = checkpoint['trainer']['stage']
aligner = BandPANAligner().cuda().eval()
aligner.load_state_dict({k[len('aligner.'):]:v for k,v in sd.items() if k.startswith('aligner.')})
entry = json.loads((root/'ablr2/sensor_sources.json').read_text())['sensors']['QB']
epsilons = torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],device='cuda')
report = {'purpose':'DIAGNOSTIC_ONLY_NO_TRAINING_NO_OFFICIAL_RECORD',
          'checkpoint':str(p.relative_to(root)), 'injection':'legacy_bicubic_border_integer_1px',
          'expected':'A(W(P,eps),LMS)-A(P,LMS)=-eps', 'splits':{}}
started = time.monotonic()
with torch.inference_mode():
    for split in ('rr','fr'):
        rows=[]
        with h5py.File(root/entry['splits'][split]['path'],'r') as f:
            for i in range(20):
                pan=torch.from_numpy(np.asarray(f['pan'][i],dtype=np.float32)).unsqueeze(0).cuda()*(2/2047)-1
                lms=torch.from_numpy(np.asarray(f['lms'][i],dtype=np.float32)).unsqueeze(0).cuda()*(2/2047)-1
                u0=aligner(pan,lms)
                ue=aligner(warp_pan(pan.expand(4,-1,-1,-1),epsilons),lms.expand(4,-1,-1,-1))
                response=ue-u0
                error=torch.linalg.vector_norm(response+epsilons,dim=1)
                gain=-(response*epsilons).sum(1)
                cross=(response[:,0]*epsilons[:,1]-response[:,1]*epsilons[:,0]).abs()
                rows.append({'index':i,'u0':u0[0].cpu().tolist(),'ue':ue.cpu().tolist(),
                    'response':response.cpu().tolist(),'equivariance_error_px':error.cpu().tolist(),
                    'signed_response_gain':gain.cpu().tolist(),'cross_axis_abs_response_px':cross.cpu().tolist()})
        result={'per_scene':rows}
        for key in ('equivariance_error_px','signed_response_gain','cross_axis_abs_response_px'):
            x=np.asarray([r[key] for r in rows])
            result[key]={'mean':float(x.mean()),'median':float(np.median(x)),
                         'p90':float(np.quantile(x,.9)),'per_direction_mean':x.mean(0).tolist()}
        report['splits'][split]=result
        print(split,{k:v for k,v in result.items() if k!='per_scene'},flush=True)
report['elapsed_seconds']=time.monotonic()-started
(root/'results_log/assets/2026-10-26_qb-aligner-response.json').write_text(json.dumps(report,indent=2)+'\n')
