"""Independent image registration proxies before/after frozen predicted shift.

FR uses only stored native LMS. RR GT reference is an explicit diagnostic
oracle, never a training/model-selection input. Neither is physical shift GT.
"""
import json
from pathlib import Path
import sys
root = Path('/home/knuvi/Desktop/song/PAN-Crafter')
sys.path.insert(0,str(root))
import cv2
import h5py
import numpy as np
import torch
from pa.warp import warp_pan
from ta2.registration import register_band
torch.set_num_threads(4)
cv2.setNumThreads(4)
entry=json.loads((root/'ablr2/sensor_sources.json').read_text())['sensors']['QB']
responses=json.loads((root/'results_log/assets/2026-10-26_qb-aligner-response.json').read_text())
report={'purpose':'DIAGNOSTIC_ONLY_NOT_PHYSICAL_DISPLACEMENT_GT',
        'checkpoint':responses['checkpoint'],'registration':{'descriptor':'gradient',
        'bound':8,'margin':16,'target_rms_threshold':1e-4},'splits':{}}
for split in ('rr','fr'):
    rows=[]
    with h5py.File(root/entry['splits'][split]['path'],'r') as f:
        for i in range(20):
            pan=np.asarray(f['pan'][i,0],np.float32)/2047.
            ref=np.asarray(f['gt' if split=='rr' else 'lms'][i],np.float32)/2047.
            u=np.asarray(responses['splits'][split]['per_scene'][i]['u0'],np.float32)
            corrected=warp_pan(torch.from_numpy(pan)[None,None],torch.from_numpy(u)[None])[0,0].numpy()
            row={'index':i,'predicted_u':u.tolist()}
            for ref_name,reference in [('rgb_mean',ref[[2,1,0]].mean(0)),('all4_mean',ref.mean(0))]:
                before=register_band(pan,reference,sigma_p=.8 if split=='rr' else 1.98,sigma_r=.8 if split=='rr' else .7)
                after=register_band(corrected,reference,sigma_p=.8 if split=='rr' else 1.98,sigma_r=.8 if split=='rr' else .7)
                row[ref_name]={'before':before,'after':after,
                    'before_norm_px':float(np.linalg.norm(before['shift'])) if before['valid'] else None,
                    'after_norm_px':float(np.linalg.norm(after['shift'])) if after['valid'] else None}
            rows.append(row)
    result={'reference':'GT_ORACLE_DIAGNOSTIC_ONLY' if split=='rr' else 'INPUT_ONLY_NATIVE_LMS',
            'sigma_p':.8 if split=='rr' else 1.98,'sigma_r':.8 if split=='rr' else .7,'per_scene':rows}
    for ref_name in ('rgb_mean','all4_mean'):
        valid=[r[ref_name] for r in rows if r[ref_name]['before']['valid'] and r[ref_name]['after']['valid']]
        b=np.asarray([r['before_norm_px'] for r in valid]);a=np.asarray([r['after_norm_px'] for r in valid])
        result[ref_name]={'valid_pairs':len(valid),'before_norm_median':float(np.median(b)),
                         'after_norm_median':float(np.median(a)),
                         'before_norm_mean':float(np.mean(b)),'after_norm_mean':float(np.mean(a)),
                         'improved_count':int(sum(a<b)),
                         'before_shift_axis_median':np.median([r['before']['shift'] for r in valid],axis=0).tolist(),
                         'after_shift_axis_median':np.median([r['after']['shift'] for r in valid],axis=0).tolist()}
    report['splits'][split]=result
    print(split,{k:v for k,v in result.items() if k!='per_scene'},flush=True)
(root/'results_log/assets/2026-10-26_qb-registration-proxy.json').write_text(json.dumps(report,indent=2)+'\n')
