"""Fixed d616740 W104 audit. CPU only, operational writes isolated in /tmp."""
import copy, hashlib, json, os, pathlib, shutil, subprocess, sys, tempfile, time, types
from unittest.mock import patch
ROOT=pathlib.Path(sys.argv[1]).resolve();OUT=pathlib.Path(sys.argv[2]).resolve()
os.chdir(ROOT);sys.path.insert(0,str(ROOT));os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PANCRAFTER_CAMPAIGN_GATES='')
import torch,yaml
torch.set_num_threads(2)
from tools import gen_pakd50_configs as G
from tools.dcr12 import common as C
from tools.smoke_cases import build,randomize_zero_params
from kdv.teacher_assets import state_hash,load_donor_aligner
from pa.model import PAModel
from pa.aligner import PANGlobalAligner
from train_pa import PATrainer
import main as Main
R=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),device='cpu',checks=[])
assert R['head']=='d616740e4b9188218f93b736394d6224b674534e'
def check(name,ok,kind='conformance',**details):
    R['checks'].append(dict(name=name,passed=bool(ok),kind=kind,**details));(OUT/'w104_verification.json').write_text(json.dumps(R,ensure_ascii=False,indent=2))
    print(('PASS ' if ok else 'FAIL ')+name+' '+json.dumps(details,ensure_ascii=False),flush=True)
def bug(name,ok,**kw):check(name,ok,kind='defect_reproduction',**kw)
arch='W104_D121';plan=[c+'@'+arch for c in ('NA0','J0','JQ','XJ','F0')]
cfgs={c:yaml.safe_load((ROOT/'config'/(G.run_name(c,1234,arch=arch)+'.yaml')).read_text()) for c in ('NA0','J0','JQ','XJ','F0')}
with tempfile.TemporaryDirectory(prefix='w104-generate-') as td:
    for rn in G.generate('s4',plan,td):
        cfg=yaml.safe_load((pathlib.Path(td)/(rn+'.yaml')).read_text());c=G.case_of(rn);k=cfg['kdv']
        check('W104_config_'+c,cfg==cfgs[c] and cfg['model_args']['hidden_size']==104 and cfg['model_args']['depth']==[1,2,1] and cfg['batch_size']==48 and cfg['num_iter']==50000 and cfg['num_warmup']==100 and k['candidate_grid_id']=='GRID1010_50K_v1' and k['experiment_branch_id']=='A104D121_T0FIX_E0_v1')
check('W104_core_order_and_7p7523h',G.priority_for('s4')==plan and abs(sum(G.reservation_for('s4',x)['reservation_h'] for x in plan)-7.752333333333)<1e-9)
for win in ('JQ','XJ','F0'):
    cases=G.confirmation_cases(win,'s4');expected=['J0@'+arch,win+'@'+arch,'J0',win]
    rr=[G.to_tag(x,3407) for x in cases]
    check('W104_confirmation_'+win,cases==expected and abs(sum(G.reservation_for('s4',x)['reservation_h'] for x in rr)-6.3426666666667)<1e-9)

T=C.load_pipe('T');cal=C.calibration();th=state_hash(T.m)
u=build(cfgs['JQ']);randomize_zero_params(u)
al,am=load_donor_aligner(f'{G.T0_ASSET_DIR}/{G.T0_TAG}',8,cfgs['JQ']['kdv']['donor']['expected_sha256'])
S=PAModel(u,al,aligner_margin=4,sampler=True)
check('W104_params_T0_A_strict_copy',sum(p.numel() for p in u.parameters())==1903624 and state_hash(al)==state_hash(T.m.aligner),params=sum(p.numel() for p in u.parameters()),A_hash=state_hash(al))
for side in (64,256,512):
    gen=torch.Generator().manual_seed(side);ms=torch.rand(1,8,side//4,side//4,generator=gen);pan=torch.rand(1,1,side,side,generator=gen);lp=torch.rand(1,1,side//4,side//4,generator=gen)
    with torch.no_grad():
        o=S(pan,ms,lp);res=u(o['pan_aligned'],lp,ms,torch.ones(1));yt=T.m(pan,ms,lp)['y']
    check('W104_forward_'+str(side)+'_residual_once',tuple(o['y'].shape)==(1,8,side,side) and torch.isfinite(o['y']).all() and torch.equal(o['y'],o['ms_base']+res) and tuple(yt.shape)==tuple(o['y'].shape))
    if side==64:small=(ms,pan,lp,yt)
    del o,res,yt
ms,pan,lp,yt=small;gt=torch.rand_like(yt)
for case in ('J0','JQ','XJ','F0','NA0'):
    m=copy.deepcopy(S).train();m.requires_grad_(True)
    if case=='F0':m.aligner.requires_grad_(False)
    if case=='NA0':m.aligner=None;m.sampler=False;m.aligner_margin=0
    tr=C.trainer_stub(case,1234,m,T.m,cal);tr.k=cfgs[case]['kdv']
    ah=state_hash(m.aligner) if m.aligner is not None else None
    loss,info=tr._step(gt,ms,lp,pan,1)
    off=info['loss_off'];up=list(m.backbone.parameters())
    offu=torch.autograd.grad(off,up,retain_graph=True,allow_unused=True) if off.requires_grad else [None]
    loss.backward();un=sum(float(p.grad.abs().sum()) for p in up if p.grad is not None)
    an=sum(float(p.grad.abs().sum()) for p in m.aligner.parameters() if p.grad is not None) if m.aligner is not None else 0
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4);opt.step()
    expected_teacher=case in ('JQ','XJ')
    check('W104_'+case+'_actual_gradient_ownership',un>0 and ((an>0)==(case in ('J0','JQ','XJ'))) and all(g is None or not g.any() for g in offu) and (info['y_t'] is not None)==expected_teacher and all(p.grad is None for p in T.m.parameters()) and state_hash(T.m)==th and (case!='F0' or ah==state_hash(m.aligner)),U_grad_l1=un,A_grad_l1=an,offset=float(off))
    if case=='NA0':check('NA0_original_PAN_no_sampler_no_Teacher_loss',m.aligner is None and not m.sampler and torch.equal(info['pan_aligned'],pan) and info['y_t'] is None and not info['delta'].any())
    del tr,m,loss,info,opt
q,x=(C.trainer_stub(c,1234,copy.deepcopy(S),T.m,cal) for c in ('JQ','XJ'))
q.rec_crit.kd_weight=0.
lq,iq=q._step(gt,ms,lp,pan,1);lx,ix=x._step(gt,ms,lp,pan,1)
check('W104_Q_beta0_equals_X',torch.equal(lq,lx),difference=float(lq-lx))
del q,x,lq,lx,iq,ix

# Use the actual main.load_data/load_model, replacing only the disk dataset
# with integer indices. DataLoader's global-generator behavior is unchanged.
class Indices(torch.utils.data.Dataset):
    def __len__(self):return 997
    def __getitem__(self,i):return i
original_import=Main.import_class;rows={};states={}
for a in ('W104_D121','W112_D123'):
    cfg=copy.deepcopy(cfgs['J0']);cfg['seed']=3407
    if a=='W112_D123':cfg['model_args'].update(hidden_size=112,depth=[1,2,3])
    cfg.update(feeder='audit.indices',num_worker=0,batch_size=16,val_batch_size=1,test_batch_size=1)
    for key in ('train_feeder_args','val_feeder_args','test_reduced_feeder_args','test_full_feeder_args'):cfg[key]={}
    args=types.SimpleNamespace(**cfg);Main.init_seed(3407)
    with patch.object(Main,'import_class',side_effect=lambda n:Indices if n=='audit.indices' else original_import(n)):
        dl=Main.load_data(args);model=Main.load_model(args)
        states[a]=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:16]
        rows[a]=next(iter(dl['train'])).tolist()
    del dl,model
bug('W104_vs_W112_same_seed_different_training_order',rows['W104_D121']!=rows['W112_D123'],seed=3407,first_batches=rows,post_model_RNG=states,note='actual main.load_data/load_model; integer dataset replaces file I/O only')

with tempfile.TemporaryDirectory(prefix='w104-init-') as td:
    tr=object.__new__(PATrainer);tr.init_dir=td;tr.args=types.SimpleNamespace(seed=1234)
    a=build(cfgs['J0']);b_cfg=copy.deepcopy(cfgs['J0']);b_cfg['model_args']['dropout']=.2;b=build(b_cfg)
    ia=tr._pair_init(a,PANGlobalAligner(8));ib=tr._pair_init(b,PANGlobalAligner(8))
    bug('W104_init_namespace_does_not_validate_full_architecture',ia['unet_init_file']==ib['unet_init_file'] and ia['unet_init_sha256_16']==ib['unet_init_sha256_16'],changed_argument='dropout 0 -> 0.2',loaded_from_file=ib.get('unet_loaded_from_file'),same_parameter_count=sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters()),note='same-width/depth tensor-compatible model change is silently accepted; no full signature is checked')
bug('W104_all_generated_configs_lack_expected_U_init_hash',all('expect_init' not in c['kdv'] for c in cfgs.values()),asset_pin=G.init_hash_for(1234,arch))

def fixture(p):
    for src in ROOT.iterdir():
        if src.name in ('.git','.agents','.codex','research_log','results_log','outputs','__pycache__'):continue
        if src.name in ('work_dir','gspread'):(p/src.name).mkdir()
        elif src.name in ('tools','config'):shutil.copytree(src,p/src.name,ignore=shutil.ignore_patterns('__pycache__'))
        else:(p/src.name).symlink_to(src,target_is_directory=src.is_dir())
    (p/'gspread/server.txt').write_text('s4');(p/'work_dir/_pakd50').mkdir();(p/'work_dir/_pakd50/ledger.json').write_text(json.dumps(G.campaign_clock()))
    (p/'work_dir/campaign_gates_enabled.txt').write_text('pakd50\n');(p/'work_dir/_pakd50_budget').mkdir();(p/'work_dir/_pakd50_budget/ledger.json').write_text(json.dumps(dict(entries={},total_gpu_hours=50)))
    (p/'tools/pakd50_unit_tests.py').write_text("print('ALL OK (separately verified)')\n");(p/'tools/pakd50_requeue.sh').write_text('#!/bin/bash\nexit 93\n')
with tempfile.TemporaryDirectory(prefix='w104-reallocate-') as td:
    p=pathlib.Path(td);fixture(p);env={**os.environ,'PYTHON':sys.executable};logs=[]
    # Resolve work_dir to this fixture before the shell's byte-for-byte check.
    z=subprocess.run([sys.executable,'-B','tools/gen_pakd50_configs.py','--server','s4','--cases',','.join(plan)],cwd=p,env=env,capture_output=True,text=True);assert z.returncode==0
    # Model old confirmation already registered, then replace it per plan §6.
    old=[G.to_tag(x,3407) for x in ('RCQ','RC0','F0')];(p/G.EXTRA_PRIORITY_FILE).write_text('\n'.join(old)+'\n')
    z=subprocess.run(['bash','tools/pakd50_reallocate.sh','--dry-run','--confirm','JQ'],cwd=p,env=env,capture_output=True,text=True,timeout=90);logs.append(z.stdout+z.stderr)
    check('W104_shell_fixture',z.returncode==0,rc=z.returncode)
    extra=(p/G.EXTRA_PRIORITY_FILE).read_text().splitlines();mand=(p/G.MANDATORY_FILE).read_text().splitlines();rr=[G.to_tag(c,3407) for c in G.confirmation_cases('JQ','s4')]
    bug('W104_old_confirmation_not_replaced_and_new_not_mandatory',all(x in extra for x in old+rr) and all(x not in mand for x in rr),extra=extra,mandatory=mand)
    bug('W104_dry_run_already_registers_confirmation',all(x in extra for x in rr),n_extra=len(extra))
    (OUT/'w104_reallocation_fixture.log').write_text('\n'.join(logs))

frozen=json.loads((OUT/'provenance_final.json').read_text());diff=[]
for f in frozen['files']:
    p=ROOT/f['path']
    if hashlib.sha256(p.read_bytes()).hexdigest()!=f['sha256']:diff.append(f['path'])
check('final_frozen_source_unchanged',not diff,changed=diff)
R['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');R['summary']={k:dict(total=sum(x['kind']==k for x in R['checks']),passed=sum(x['kind']==k and x['passed'] for x in R['checks'])) for k in sorted({x['kind'] for x in R['checks']})}
(OUT/'w104_verification.json').write_text(json.dumps(R,ensure_ascii=False,indent=2));print(json.dumps(R['summary']),flush=True)
sys.exit(0 if all(x['passed'] for x in R['checks']) else 1)
