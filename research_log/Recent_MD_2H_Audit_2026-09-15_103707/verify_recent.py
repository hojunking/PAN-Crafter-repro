"""Read-only audit of five MDs in the fixed 08:37:07--10:37:07 KST window.
Real CPU math checks; all unit/shell/gate writes isolated under /tmp.
Usage: python -B verify_recent.py REPO OUTPUT_DIR
"""
import ast,contextlib,copy,hashlib,io,json,os,pathlib,runpy,shutil,subprocess,sys,tempfile,time,types
from unittest.mock import patch
ROOT=pathlib.Path(sys.argv[1]).resolve(); OUT=pathlib.Path(sys.argv[2]).resolve(); OUT.mkdir(exist_ok=True,parents=True)
os.chdir(ROOT); sys.path.insert(0,str(ROOT)); os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PANCRAFTER_CAMPAIGN_GATES='')
import numpy as np,pandas as pd,torch,yaml
torch.set_num_threads(2)
from tools import gen_pakd50_configs as G
from tools.dcr12 import common as C,d01,d02,d03,d04,report,results
from train_kdv import KDVTrainer
from kdv.registry import resolve
from kdv.teacher_assets import state_hash
R={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'checks':[],'device':'cpu'}
def check(name,ok,kind='conformance',**details):
    R['checks'].append(dict(name=name,passed=bool(ok),kind=kind,**details)); (OUT/'verification.json').write_text(json.dumps(R,ensure_ascii=False,indent=2))
    print(('PASS ' if ok else 'FAIL ')+name+' '+json.dumps(details,ensure_ascii=False),flush=True)
def bug(name,ok,**details): check(name,ok,kind='defect_reproduction',**details)

def fixture(p,srv='s1'):
    for src in ROOT.iterdir():
        if src.name in ('.git','.agents','.codex','research_log','results_log','outputs','__pycache__'): continue
        if src.name in ('work_dir','gspread'): (p/src.name).mkdir()
        elif src.name in ('tools','config'): shutil.copytree(src,p/src.name,ignore=shutil.ignore_patterns('__pycache__'))
        else: (p/src.name).symlink_to(src,target_is_directory=src.is_dir())
    (p/'gspread/server.txt').write_text(srv); (p/'work_dir/_pakd50').mkdir()
    (p/'work_dir/_pakd50/ledger.json').write_text(json.dumps(G.campaign_clock()))
    (p/'work_dir/campaign_gates_enabled.txt').write_text('pakd50\n')
    (p/'work_dir/_pakd50_budget').mkdir(); (p/'work_dir/_pakd50_budget/ledger.json').write_text(json.dumps(dict(entries={},total_gpu_hours=50)))

with tempfile.TemporaryDirectory(prefix='recent-md-pakd-unit-') as td:
    p=pathlib.Path(td);fixture(p)
    z=subprocess.run([sys.executable,'-B',str(p/'tools/pakd50_unit_tests.py')],cwd=p,text=True,capture_output=True,timeout=240)
    (OUT/'pakd_unit.log').write_text(z.stdout+z.stderr)
    check('PAKD_unit_suite_isolated',z.returncode==0,ok_count=sum(l.startswith('  OK ') for l in z.stdout.splitlines()),rc=z.returncode)

with tempfile.TemporaryDirectory(prefix='recent-md-dcr-unit-') as td:
    log=io.StringIO(); rc=0
    with patch.object(C,'CAMP',td),patch.object(C,'FIG',os.path.join(td,'figures')),contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try: runpy.run_path(str(ROOT/'tools/dcr12_unit_tests.py'),run_name='__main__')
        except SystemExit as e: rc=e.code
    (OUT/'dcr13_unit.log').write_text(log.getvalue())
    check('DCR_unit_suite_isolated_output',rc==0,ok_count=sum(l.startswith('  OK ') for l in log.getvalue().splitlines()),rc=rc)

plan=['J_R3_NOEDGE','J_N0_EDGE','LF0','LFQ','LFX']
with tempfile.TemporaryDirectory(prefix='recent-md-config-') as td:
    for rn in G.generate('s3',plan,td):
        a=yaml.safe_load((ROOT/'config'/(rn+'.yaml')).read_text()); b=yaml.safe_load((pathlib.Path(td)/(rn+'.yaml')).read_text()); k=a['kdv']; sp=resolve(k)
        check('s3_config_'+G.case_of(rn),a==b and a['seed']==2026 and a['num_iter']==50000 and a['batch_size']==48 and a['num_warmup']==100 and k['candidate_grid_id']=='GRID1010_50K_v1')
check('s3_explicit_queue_and_reservations',G.priority_for('s3')==plan and G.mandatory_for('s3')==plan and abs(sum(G.reservation_for('s3',x)['reservation_h'] for x in plan)-7.994333333333)<1e-9)
bundles={'LFQ':['JQ','LF0','LFQ'],'LFX':['XJ','LF0','LFX'],'J_N0_EDGE':['J0','XJ','J_N0_EDGE'],'J_R3_NOEDGE':['J0','JR','J_R3_NOEDGE'],'XJ':['J0','JR','XJ'],'JQ':['J0','XJ','JQ']}
check('s3_six_confirmation_bundles',G.CONFIRM_SEED['s3']==4321 and all(G.confirmation_cases(c,'s3')==v for c,v in bundles.items()))
check('s3_confirmation_budget',all(abs(sum(G.reservation_for('s3',G.run_name(x,4321))['reservation_h'] for x in xs)-4.922)<1e-9 for xs in bundles.values()))
cal=C.calibration(); T=C.load_pipe('T'); template=copy.deepcopy(T.m)
from safetensors.torch import load_file
sd=load_file(str(ROOT/'work_dir'/G.pilot_run()/'candidates/step-4040/model.safetensors'))
template.backbone.load_state_dict({k[9:]:v for k,v in sd.items() if k.startswith('backbone.')})
inputs=C.load_patches([13,79]); gt,ms,lpan,pan=inputs
def make(case):
    m=copy.deepcopy(template).train().requires_grad_(True)
    if case=='FQ': m.aligner.requires_grad_(False)
    return C.trainer_stub(case,2026,m,T.m,cal,seed_gen=4026)
def grad(loss,params): return C.flat(torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True),params)
for case in ('JK0','LFX'):
    tr=make(case); total,info=tr._step(*inputs,24999); y=info['y']; yt=info['y_t']; et=(yt-gt).abs().mean(1); es=(y-gt).abs().mean(1)
    d=(et/(et+cal['tau_R'])).detach(); a=((es-et).clamp_min(0)/(es+1e-6)).detach(); lh=((1+d)*es).mean()
    lk=(.1*(1-d)*a*(y-yt).abs().mean(1)).mean() if case=='JK0' else y.sum()*0
    from pa.losses import output_edge_loss
    le=cal['lambda_E']*output_edge_loss(y,gt); lo=1e-4*info['loss_off']; manual=lh+lk+le+lo
    ap=list(tr.M.aligner.parameters()); bp=list(tr.M.backbone.parameters()); ga=grad(lh+le+lo,ap); gu=grad(manual,bp)
    total.backward(retain_graph=tr._routed)
    if tr._routed:tr._apply_routing(info,tr.M)
    ea=float((torch.cat([p.grad.flatten() for p in ap])-ga).abs().max()); eu=float((torch.cat([p.grad.flatten() for p in bp])-gu).abs().max())
    check(case+'_actual_loss_A_U_gradients',abs(float(total-manual))<1e-7 and ea<2e-6 and eu<2e-6,loss_error=abs(float(total-manual)),A_error=ea,U_error=eu,soft=float(lk))
    del tr,total,info,manual,ga,gu
for lf,joint in [('LF0','J0'),('LFQ','JQ'),('LFX','XJ')]:
    a,b=make(lf),make(joint)
    opta=torch.optim.AdamW([dict(params=a.M.backbone.parameters(),lr=1e-4),dict(params=a.M.aligner.parameters(),lr=1e-5)],weight_decay=.01)
    optb=torch.optim.AdamW([dict(params=b.M.backbone.parameters(),lr=1e-4),dict(params=b.M.aligner.parameters(),lr=1e-5)],weight_decay=.01)
    ok=True
    for step in (0,24999):
        ta,ia=a._step(*inputs,step);tb,ib=b._step(*inputs,step)
        ta.backward();tb.backward();opta.step();optb.step();opta.zero_grad(set_to_none=True);optb.zero_grad(set_to_none=True)
        ok=ok and torch.equal(ta,tb) and state_hash(a.M)==state_hash(b.M) and torch.equal(a.gen.get_state(),b.gen.get_state())
    check(lf+'_joint_prefix_local_two_update_identity',ok,joint=joint,indices=[0,24999],note='bounded CPU state-transition check, not a full 25K/GPU reproducibility claim')
    for step in (25000,25001):
        ah=state_hash(a.M.aligner);uh=state_hash(a.M.backbone);a.M.aligner.requires_grad_(a.aligner_active(step));ta,ia=a._step(*inputs,step);ta.backward();opta.step();opta.zero_grad(set_to_none=True)
        check(lf+'_freeze_'+str(step),state_hash(a.M.aligner)==ah and state_hash(a.M.backbone)!=uh and float(ia['loss_off'])==0 and all(p.grad is None for p in a.M.aligner.parameters()))
    del a,b,opta,optb,ta,tb,ia,ib

# Fixed boundaries are not fixed reference sample labels: B1 migration.
sm=pd.DataFrame([dict(pipeline='B0_S1234_best_hqnr',seed=1234,panel='CAL',sample_id=1,C=.1,R_plain=.1),dict(pipeline='B1_S1234_best_hqnr',seed=1234,panel='CAL',sample_id=1,C=.9,R_plain=.9)])
th={'T0':{'tC':.5,'tR':.5},'S1234':{'tC':.5,'tR':.5,'reference':'B0_S1234_best_hqnr'}}
labels=d01.assign_cells(sm,th).cell_ref.tolist();bug('D01_reference_sample_migrates_A_to_D',labels==['A','D'],labels=labels)

# Adversarial sufficient-looking utility samples: gate must still enforce the
# metadata/support/context restrictions mandated by the plan.
def utility(panels=('DISC','CONF'),steps=(25250,45450),groups=8):
    rows=[]
    for panel in panels:
        for step in steps:
            for cell in 'ABCD':
                for ep in range(8): rows.append(dict(panel=panel,step=step,cell=cell,episode=ep,U=-.002 if cell=='B' else .002,group_key='g'+str(ep%groups)))
    return pd.DataFrame(rows)
thr={'T0':dict(degenerate_metric=False,C_cv=.3)}
g=d04.gate(utility(),thr,dict(independence='patch_only'))
bug('D04_patch_only_metadata_does_not_close_gate',g['passed'],gate=g)
g=d04.gate(utility(steps=(25250,)),thr,None);bug('D04_one_checkpoint_still_opens',g['passed'],contexts=g['conditions']['3_harmful_cell_reproduced']['detail']['B']['n_contexts'])
g=d04.gate(utility(panels=('CONF',)),thr,None);bug('D04_no_DISC_still_opens',g['passed'])
g=d04.gate(utility(groups=2),thr,None);bug('D04_only_two_groups_still_opens',g['passed'],groups=2)
df=utility()
for panel,c,v in [('DISC','A',-.001),('DISC','B',-.004),('CONF','A',-.004),('CONF','B',-.001)]:df.loc[(df.panel==panel)&(df.cell==c),'U']=v
g=d04.gate(df,thr,None);bug('D04_zstar_selected_using_CONF_instead_of_DISC',g['z_star']=='A',DISC_worst='B',selected=g['z_star'])
sup,qry=d04.episode_split(list(range(128)),np.arange(128),np.random.RandomState(7)); overlap=set(map(C.group_of,sup))&set(map(C.group_of,qry))
bug('D04_episode_support_query_share_source_groups',bool(overlap),overlap=sorted(overlap),patch_disjoint=not(set(sup)&set(qry)))

# Real measured per-sample D03 on identical Teacher/Student states, zero noise
# floor: zero norm must be NA, not an artificial cosine zero.
m=d03.open_A(T);pars=list(m.aligner.parameters())
with torch.no_grad():yt=m(pan[:1],ms[:1],lpan[:1])['y'].clone()
row=d03.per_sample(T,m,pars,pan[:1],ms[:1],lpan[:1],gt[:1],yt,cal,0.)
bug('D03_zero_floor_zero_soft_norm_not_marked_NA',row['norm_LK']==0 and 'LK' not in row['na_zero_grad'] and row['cos_gT_gK']==0,norm_LK=row['norm_LK'],na=row['na_zero_grad'],cos=row['cos_gT_gK'])

# Saved run validity is checked only by files, and stage resume accepts invalid
# contents. Use the actual CLI checker without invoking any stage.
ns=runpy.run_path(str(ROOT/'tools/dcr12.py'),run_name='audit_import')
with tempfile.TemporaryDirectory(prefix='recent-md-validation-') as td:
    p=pathlib.Path(td)
    with patch.object(C,'CAMP',td):
        for name in ns['REQUIRED']['d00']:(p/name).write_text('{}')
        (p/'d00_smoke.json').write_text(json.dumps({'implementation_invalid':True,'hard_failures':['soft_routing']}))
        bug('D00_resume_checker_accepts_invalid_smoke',ns['check']('d00'))
        for name in ns['REQUIRED']['d03']:(p/name).write_text('{}')
        (p/'d03_summary.json').write_text(json.dumps({'routing_checks':[{'ok':False}]}))
        bug('D03_resume_checker_accepts_routing_failure',ns['check']('d03'))
    with patch.object(C,'run_dir',return_value=td):
        (p/'results').mkdir();(p/'results/reduced_best_hqnr.mat').touch();(p/'results/full_best_hqnr.mat').touch()
        bug('DCR_complete_accepts_empty_exports_without_exit_last',C.run_complete('B1',777))
        bug('DCR_last_missing_metadata_assumes_50000',C.ckpt_step('B1',777,'last')==50000)

v=report.verdicts({}, {}, {}, pd.DataFrame(),{'passed':True})
bug('report_ROUTING_candidate_without_B2_B3_results',any(x.startswith('QUADRANT_ROUTING_CANDIDATE') for x in v['labels']),labels=v['labels'])
v=report.verdicts({'T0|DISC':{'spearman_C_R':{'rho':.5,'ci95':[.3,.7]}}},{},{},pd.DataFrame(),None)
bug('report_ROBUSTNESS_label_without_robustness_or_D04_evidence',any(x.startswith('ROBUSTNESS_ONLY') for x in v['labels']),labels=v['labels'])

# Missing optional candidates should not prevent a mandatory report, but D04
# raises before producing a not-run result; no training/micro-update is invoked.
with tempfile.TemporaryDirectory(prefix='recent-md-d04-missing-') as td:
    p=pathlib.Path(td);pd.DataFrame([dict(pipeline='T0',sample_id=1)]).to_csv(p/'sample_metrics.csv',index=False);(p/'quadrant_thresholds.json').write_text(json.dumps(thr))
    with patch.object(C,'CAMP',td),patch.object(C,'FIG',str(p/'figures')),patch.object(C,'make_panels',return_value={}),patch.object(C,'run_complete',return_value=False):
        try:d04.main();err=None
        except FileNotFoundError as e:err=str(e)
    bug('D04_no_candidates_raises_instead_of_not_run',bool(err) and 'micro_update_utility.csv' in err,error=err)

# s3 uses the same operational code previously audited. Run the actual dry-run
# and confirmation script in a fake s3 to verify current behavior, not old text.
with tempfile.TemporaryDirectory(prefix='recent-md-s3-reallocate-') as td:
    p=pathlib.Path(td);fixture(p,'s3');(p/'tools/pakd50_unit_tests.py').write_text("print('ALL OK (unit suite separately verified)')\n")
    (p/'tools/pakd50_requeue.sh').write_text('#!/bin/bash\nexit 93\n');env={**os.environ,'PYTHON':sys.executable}
    z=subprocess.run([sys.executable,'-B','tools/gen_pakd50_configs.py','--server','s3','--cases',','.join(plan)],cwd=p,env=env,capture_output=True,text=True);assert z.returncode==0
    logs=[]
    for candidate in ('LFX','J_N0_EDGE'):
        z=subprocess.run(['bash','tools/pakd50_reallocate.sh','--dry-run','--confirm',candidate],cwd=p,env=env,capture_output=True,text=True,timeout=90);logs.append(z.stdout+z.stderr);check('s3_shell_fixture_'+candidate,z.returncode==0,rc=z.returncode)
        if candidate=='LFX':
            extra=(p/G.EXTRA_PRIORITY_FILE).read_text().splitlines();mand=(p/G.MANDATORY_FILE).read_text().splitlines();conf=[G.run_name(c,4321) for c in bundles['LFX']]
            bug('s3_dry_run_registers_live_confirmations_and_loses_mandatory',all(x in extra and x not in mand for x in conf),extra=extra,mandatory=mand)
    extra=(p/G.EXTRA_PRIORITY_FILE).read_text().splitlines();bug('s3_repeated_WIN_exceeds_3_runs',len(extra)==5,extra=extra)
    (OUT/'s3_reallocation_fixture.log').write_text('\n'.join(logs))
todo,dropped=G.schedule(True,lambda x:False,9,lambda x:G.reservation_for('s3',x)['reservation_h'],priority=plan,extra=[G.run_name(x,4321) for x in bundles['LFX']])
bug('s3_admission_spends_core_budget_without_reserving_confirmation_pair',todo==plan and len(dropped)==3,remaining_h=9,core_h=7.9943333333,required_with_confirmation_h=12.9163333333,todo=todo,dropped=dropped)

R['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');R['summary']={k:dict(total=sum(x['kind']==k for x in R['checks']),passed=sum(x['kind']==k and x['passed'] for x in R['checks'])) for k in sorted({x['kind'] for x in R['checks']})}
(OUT/'verification.json').write_text(json.dumps(R,ensure_ascii=False,indent=2));print(json.dumps(R['summary']),flush=True)
sys.exit(0 if all(x['passed'] for x in R['checks']) else 1)
