"""Resolve whether CPU selector reconstruction differs from actual trainer inference."""
import os,sys,types,json,pathlib
from audit_common import *
os.environ.update(ENV);os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import torch,yaml,numpy as np
from scipy.io import loadmat
from tools.eval_fr_paperset import build
from main import import_class
from train_kdv import KDVTrainer
from kdv.teacher_assets import load_run_model
from kdv.forward import kdv_forward
w=ROOT/'work_dir/PAKD50_JQ_W104_D121_WV3_T0_S1234_FRESH50_v2';cfg=yaml.safe_load((w/'meta/config.yaml').read_text());torch.set_num_threads(2)
m,fwd,how=build(cfg,str(w),'candidates/step-27270');m.eval();mt,_=load_run_model(str(w),'best_hqnr',import_class(cfg['model']));mt.eval();ds=import_class(cfg['feeder'])(**cfg['test_reduced_feeder_args']);gt,lms,ms,lp,pan=[x.unsqueeze(0) for x in ds[0]]
tr=KDVTrainer.__new__(KDVTrainer);tr.model=mt;tr.accelerator=types.SimpleNamespace(unwrap_model=lambda m:m,device=torch.device('cpu'));tr.weight_dtype=torch.float32
with torch.no_grad():
 y=fwd(pan,lp,ms,lms);ot=tr._infer(pan,lp,ms);out=kdv_forward(mt,None,pan,ms,lp,teacher_needed=False,share_correction=False,aligner_live=False)
old=loadmat(w/'results/reduced_best_hqnr.mat');rec=dict(passed=torch.equal(y,ot['y']) and torch.equal(y,out['y']),selector_vs_trainer_max=float((y-ot['y']).abs().max()),selector_vs_training_forward_max=float((y-out['y']).abs().max()),cpu_shift=ot['delta'].tolist(),export_gpu_shift=old['delta'][0].tolist(),shift_max_difference=float(np.max(np.abs(ot['delta'].numpy()[0]-old['delta'][0]))),note='Same CPU inputs/checkpoint: independent load_run_model + real PATrainer._infer and kdv_forward versus selector helper build. Cross-device export comparison remains a separate numerical discrepancy, not classified as proven selector algorithm error.')
(OUT/'inference_paths.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
