import os,sys,pathlib,json,tempfile,shutil,importlib.util,hashlib,datetime
from audit_common import *
os.environ.update(ENV);os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import torch,numpy as np
from scipy.io import loadmat
from tools import qrecon24_select as S
from train_pa import _sha_tensors
from kdv.teacher_assets import sha256_file
R=[]
def rec(name,ok,kind='conformance',**kw):
 R.append(dict(name=name,passed=bool(ok),kind=kind,**kw));(OUT/'selector_live_results.json').write_text(json.dumps(R,indent=2,ensure_ascii=False));print(name,ok,kw,flush=True)
torch.set_num_threads(2)
# Save/read actual running checkpoint metadata; do not read weights being written.
w=ROOT/'work_dir/PAKD50_QRC24_S1_G22_W104_D121_WV3_T0_S1234_FRESH50_v1';cp=w/'candidates/step-1010';kd=json.loads((w/'kdv_config_resolved.json').read_text());init=json.loads((w/'initialization_hashes.json').read_text());ih=json.loads((w/'init_and_teacher_hashes.json').read_text());op=torch.load(cp/'optimizer.bin',map_location='cpu',weights_only=False);groups=op['param_groups'];sc=torch.load(cp/'scheduler.bin',map_location='cpu',weights_only=False)
rec('live_s1_optimizer_init_teacher',len(groups)==2 and abs(groups[1]['lr']/groups[0]['lr']-.01)<1e-15 and all(tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==.01 for g in groups) and ih['aligner_hash0']=='493aca181f93ae68' and ih['teacher']['file_sha256']=='16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32' and _sha_tensors(torch.load(init['unet_init_file'],map_location='cpu',weights_only=False))==init['unet_init_sha256_16'],groups=[{k:v for k,v in g.items() if k!='params'}|{'n_params':len(g['params'])} for g in groups],scheduler=sc,init_sha16=init['unet_init_sha256_16'],release=(w/'meta/git_commit.txt').read_text().strip())
prior=ROOT/'work_dir/PAKD50_JQ_W104_D121_WV3_T0_S1234_FRESH50_v2';bm=json.loads((prior/'best_hqnr_meta.json').read_text());step=bm['step']
with tempfile.TemporaryDirectory(prefix='qrc24-official-rr-') as td:
 p=pathlib.Path(td);(p/'meta').mkdir();shutil.copy2(prior/'meta/config.yaml',p/'meta/config.yaml');c=p/'candidates'/f'step-{step}';c.mkdir(parents=True);shutil.copy2(prior/f'candidates/step-{step}/model.safetensors',c/'model.safetensors')
 metrics,how=S.official_rr(prior.name,str(p),step,'cpu');new=loadmat(p/'results'/f'reduced_candidate_step-{step}.mat')['sr'];old=loadmat(prior/'results/reduced_best_hqnr.mat')['sr'];mx=float(np.max(np.abs(new-old)));rmse=float(np.sqrt(np.mean((new-old)**2)))
 spec=importlib.util.spec_from_file_location('gu_audit',ROOT/'gspread/gspread_upload.py');gu=importlib.util.module_from_spec(spec);spec.loader.exec_module(gu);ref=gu._rr(str(prior/'results/reduced_best_hqnr.mat'),'wv3');ref['q8']=ref['q2n'];diff={k:abs(float(metrics[k])-float(ref[k])) for k in ('scc','ergas','psnr','sam','q8','ssim')}
 rec('official_RR_W104_export_equivalence',mx<.1 and max(diff.values())<1e-4,checkpoint_step=step,max_pixel_error_DN=mx,pixel_rmse_DN=rmse,metrics={k:metrics[k] for k in diff},metric_abs_differences=diff,device='cpu',reference_device='original completed run export')
 # Stale cached .mat remains accepted as official when checkpoint bytes change.
 ck=c/'model.safetensors';sha0=sha256_file(ck);ck.write_bytes(b'not-a-valid-checkpoint-audit-only');sha1=sha256_file(ck);again,how2=S.official_rr(prior.name,str(p),step,'cpu')
 rec('official_cache_unbound_to_checkpoint',sha0!=sha1 and again==metrics and how2=='official','defect_reproduction',note='Only /tmp copy replaced; cached .mat bypassed loading a now invalid checkpoint.')
# Missing numeric optimizer fields from resolved training manifest.
rec('resolved_optimizer_contract_incomplete',all(k not in kd['training'] for k in ('betas','eps','tf32','gradient_accumulation_steps','parameter_groups','cosine_min_lr')),'defect_reproduction',resolved_training=kd['training'],note='Actual checkpoint groups correct, but required resolved manifest fields absent.')
print('SUMMARY',len(R),sum(r['passed'] for r in R))
