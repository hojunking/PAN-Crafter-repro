"""Diagnostic only: replay frozen QB snapshots with predicted vs zero PAN shift.

Uses the archived Sensor5 v1 implementation. No training, official artifacts,
references, checkpoint contents or sheet rows are modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time

p = argparse.ArgumentParser()
p.add_argument('--root', default='/home/knuvi/Desktop/song/PAN-Crafter')
p.add_argument('--archived-code', default='/tmp/panda_sensor5_v1_qb_audit')
p.add_argument('--output', default='results_log/assets/2026-10-26_qb-shift-counterfactual.json')
p.add_argument('--splits', nargs='+', choices=('rr','fr'), default=['rr','fr'])
p.add_argument('--conditions', nargs='+', choices=('predicted','zero'), default=['predicted','zero'])
p.add_argument('--reference', choices=('original','shifted'), default='original')
args = p.parse_args()
root = Path(args.root).resolve()
sys.path[:0] = [str(Path(args.archived_code).resolve()), str(root)]

import h5py
import numpy as np
import torch
from panda_sensor5.aligner import BandPANAligner
from panda_sensor5.base_reconstruction import (BandReconstructionSpec,
    build_reconstructor as build_teacher, base_from_batch, reconstruct_with_base)
from panda_sensor5.reconstruction import StudentReconstructionSpec, build_reconstructor as build_student
from panda_midpoint.resampling import upsample_lrms_after_augmentation
from panda_sensor5.warp import warp_pan as warp_pan_lanczos4
from pa.warp import warp_pan
from ta2.evaluation import rr_scene, fr_scene, evaluator_identity
from tools.metrics.eval_fr import load_dlpan

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
campaign = root / 'work_dir/panda_sensor5/PANDA_S13_SENSOR5_STAGE12_20261026_v1/s1/QB'
entry = json.loads((root/'ablr2/sensor_sources.json').read_text())['sensors']['QB']
wald = load_dlpan(str(root.parent/'DLPan-Toolbox'))
report = {'purpose': 'DIAGNOSTIC_ONLY_NOT_OFFICIAL_NOT_TRAINED_ZERO_SHIFT_MODEL',
          'archive': str(Path(args.archived_code).resolve()), 'torch': torch.__version__,
          'reference_policy': ('ORIGINAL_UNSHIFTED_REFERENCES' if args.reference=='original' else
                               'SHIFTED_PAN_REFERENCE_DIAGNOSTIC_INELIGIBLE'),
          'evaluator_identity': evaluator_identity(), 'conditions': {}}
started = time.monotonic()

for role, run, step in [('teacher', '*G4*', 38000), ('student', '*QB_BASE*', 12000)]:
    run_dir = next(campaign.glob('runs/'+run+'/attempt_01'))
    checkpoint = run_dir/'models'/f'step_{step:07d}.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    sd = state['trainer']['stage']
    aligner = BandPANAligner().eval().cuda()
    aligner.load_state_dict({k[len('aligner.'):]:v for k,v in sd.items() if k.startswith('aligner.')}, strict=True)
    if role == 'teacher':
        recon = build_teacher(BandReconstructionSpec(4,112,(1,2,3),'P0','BICUBIC_MS',1),seed=0,role='T')
    else:
        recon = build_student(StudentReconstructionSpec(4,112,(1,2,3),'P0','BICUBIC_MS',2),seed=0,role='S',sampling_variant='UBIL',input_variant='CORRECTED_DIFF')
    recon.load_state_dict({k[len(role)+1:]:v for k,v in sd.items() if k.startswith(role+'.')}, strict=True)
    recon.eval().cuda()
    official = json.loads((run_dir/f'evaluation_{step:07d}.json').read_text())
    for condition in args.conditions:
        key = role+'_'+condition
        result = {'checkpoint': str(checkpoint.relative_to(root)), 'step': step,
                  'file_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  'warp': 'bicubic_border' if role=='teacher' else 'lanczos4_reflection',
                  'scale': 1 if condition=='predicted' else 0, 'splits': {}}
        with torch.inference_mode():
            for split in args.splits:
                rows = []
                with h5py.File(root/entry['splits'][split]['path'],'r') as f:
                    for i in range(20):
                        arrays = {name:np.asarray(f[name][i],dtype=np.float32) for name in ('pan','ms','lms')+(('gt',) if split=='rr' else ())}
                        tensors = {name:torch.from_numpy(value.copy()).unsqueeze(0).cuda()*(2./2047)-1 for name,value in arrays.items()}
                        batch = SimpleNamespace(pan=tensors['pan'],ms=tensors['ms'],native_lms=tensors['lms'])
                        predicted = aligner(batch.pan,batch.native_lms)
                        applied = predicted if condition=='predicted' else torch.zeros_like(predicted)
                        corrected = (warp_pan if role=='teacher' else warp_pan_lanczos4)(batch.pan,applied)
                        inputs = corrected if role=='teacher' else torch.cat((corrected,corrected-batch.pan),dim=1)
                        base = upsample_lrms_after_augmentation(batch.ms,((False,False,0),))
                        out = reconstruct_with_base(recon,inputs,base,batch.ms)['y']
                        raw = ((out[0].float()+1)*(2047/2)).cpu().numpy()
                        reference_pan = arrays['pan']
                        if split=='fr' and args.reference=='shifted':
                            # Fixed donor-predicted displacement, including a zero-input condition.
                            shifted_reference = (warp_pan if role=='teacher' else warp_pan_lanczos4)(batch.pan,predicted)
                            reference_pan = ((shifted_reference[0]+1)*(2047/2)).cpu().numpy()
                        metrics = (rr_scene(raw,arrays['gt'],'QB') if split=='rr' else
                                   fr_scene(raw,reference_pan,arrays['lms'],arrays['ms'],'QB',wald))
                        names = ('ergas','scc') if split=='rr' else ('hqnr','d_s','d_lambda')
                        rows.append({'index':i,'predicted_dy_dx':predicted[0].cpu().tolist(),
                                     'applied_dy_dx':applied[0].cpu().tolist(),
                                     'metrics':{n:float(metrics[n]) for n in names}})
                means = {n:float(np.mean([r['metrics'][n] for r in rows])) for n in names}
                original = official[split]['metrics']
                mapping = {'ergas':'ERGAS','scc':'SCC','hqnr':'HQNR','d_s':'D_s','d_lambda':'D_lambda'}
                result['splits'][split]={'mean':means,'per_scene':rows,
                    'delta_from_official_native':{n:means[n]-original[mapping[n]] for n in names}}
        report['conditions'][key] = result
        report['elapsed_seconds'] = time.monotonic()-started
        (root/args.output).write_text(json.dumps(report,indent=2)+'\n')
        print(key,{s:r['mean'] for s,r in result['splits'].items()},flush=True)
    del recon,aligner,state,sd
    torch.cuda.empty_cache()
