"""QEDGE9 audit, fixed commit 433ad67. CPU math and isolated operational fixtures.
python -B verify_qedge9.py REPO OUTPUT_DIR
"""
import ast,contextlib,copy,hashlib,io,json,os,pathlib,random,runpy,shutil,subprocess,sys,tempfile,time,types
from unittest.mock import patch
ROOT=pathlib.Path(sys.argv[1]).resolve();OUT=pathlib.Path(sys.argv[2]).resolve();OUT.mkdir(parents=True,exist_ok=True)
os.chdir(ROOT);sys.path.insert(0,str(ROOT));os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PANCRAFTER_CAMPAIGN_GATES='')
import torch,numpy as np,yaml,h5py
torch.set_num_threads(2)
from tools import gen_pakd50_configs as G,qedge9_cue as Q
from kdv import edge_gate as EG
from kdv.edge_gate import EdgeGate
from kdv.teacher_assets import state_hash
from kdv.registry import resolve
from train_kdv import KDVTrainer
from kdv.losses_rec import GTAnchoredReconstructionKD
from pa.model import PAModel
from pa.losses import output_edge_loss_per_sample
from tools.smoke_cases import build,randomize_zero_params
R=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),device='cpu',checks=[])
assert R['head']=='433ad67f4ae0da7516e3c1dcaa6845af1ec1e837'
def check(name,ok,kind='conformance',**details):
    R['checks'].append(dict(name=name,passed=bool(ok),kind=kind,**details));(OUT/'verification.json').write_text(json.dumps(R,ensure_ascii=False,indent=2))
    print(('PASS ' if ok else 'FAIL ')+name+' '+json.dumps(details,ensure_ascii=False),flush=True)
def bug(name,ok,**kw):check(name,ok,kind='defect_reproduction',**kw)
def fixture(p,srv):
    for src in ROOT.iterdir():
        if src.name in ('.git','.agents','.codex','research_log','results_log','outputs','__pycache__'):continue
        if src.name in ('work_dir','gspread'):(p/src.name).mkdir()
        elif src.name in ('tools','config'):shutil.copytree(src,p/src.name,ignore=shutil.ignore_patterns('__pycache__'))
        else:(p/src.name).symlink_to(src,target_is_directory=src.is_dir())
    (p/'gspread/server.txt').write_text(srv);(p/'work_dir/_pakd50').mkdir();(p/'work_dir/_pakd50/ledger.json').write_text(json.dumps(G.campaign_clock()))
    (p/'work_dir/campaign_gates_enabled.txt').write_text('pakd50\n');(p/'work_dir/_pakd50_budget').mkdir();(p/G.LEDGER).write_text(json.dumps(dict(entries={},total_gpu_hours=50)))
    # Config byte checks must resolve work_dir to the isolated repository.
    for f in (p/'config').glob('*.yaml'):
        f.write_text(f.read_text().replace(str(ROOT)+'/work_dir/',str(p)+'/work_dir/'))

with tempfile.TemporaryDirectory(prefix='qedge9-units-') as td:
    p=pathlib.Path(td);fixture(p,'s1')
    z=subprocess.run([sys.executable,'-B','tools/pakd50_unit_tests.py'],cwd=p,text=True,capture_output=True,timeout=240)
    (OUT/'existing_units.log').write_text(z.stdout+z.stderr)
    check('existing_unit_suite_isolated',z.returncode==0,rc=z.returncode,count=sum(l.startswith('  OK ') for l in z.stdout.splitlines()))

# All nine new configurations and their independent plan constants.
arch='W104_D121';cfgs={}
with tempfile.TemporaryDirectory(prefix='qedge9-config-') as td:
    for srv,items in [('s5',G.priority_for('s5')),('s4',['QE50@'+arch,'QEC@'+arch,'QES@'+arch])]:
        for rn in G.generate(srv,items,td):
            c=yaml.safe_load((ROOT/'config'/(rn+'.yaml')).read_text());g=yaml.safe_load((pathlib.Path(td)/(rn+'.yaml')).read_text());k=c['kdv'];case=G.case_of(rn);cfgs[rn]=c
            check('config_'+case+'_S'+str(c['seed']),c==g and c['model_args']['hidden_size']==104 and c['model_args']['depth']==[1,2,1] and c['num_iter']==50000 and c['batch_size']==48 and c['num_warmup']==100 and float(c['learning_rate'])==1e-4 and k['aligner_lr']==1e-5 and k['teacher']['bridge'] and k['candidate_grid_id']=='GRID1010_50K_v1' and k['campaign_id']==G.QEDGE9_CAMPAIGN_ID and k['budget']['required'] and 'training_deadline' not in k['budget'] and k['budget']['ledger']==G.QEDGE9_LEDGER and bool(c['train_feeder_args'].get('return_meta'))==(case in ('QE50','QES')))
check('s5_order_two_seeds',[(G.case_of(x),G.seed_of(G.to_tag(x,2026))) for x in G.priority_for('s5')]==[(c,s) for s in (2026,777) for c in ('J0','JQ','QE50')])
check('s4_residual_plus_gate_order',[G.case_of(x) for x in G.priority_for('s4')][2:]==['JQ','XJ','F0','QE50','QEC','QES'])
check('reservation_arithmetic',abs(sum(G.reservation_for('s5',x)['reservation_h'] for x in G.priority_for('s5'))-10.262)<1e-9 and abs(sum(G.reservation_for('s4',x)['reservation_h'] for x in G.priority_for('s4')[2:])-10.163)<1e-9)

man,z=EG.read_asset(G.QEDGE9_CUE_ASSET);arrays={k:z[k].copy() for k in z.files};N=man['dataset']['n_train'];cal_ids,cal_sha=Q.calibration_ids(N)
idx,rot=arrays['index'],arrays['rot'];cm=arrays['calib_mask'];qv=arrays['q'];low=arrays['gate_low_q'];shuf=arrays['gate_shuffle']
st,dec=EG.strata_of(arrays['e_roi32'],rot,man['e_roi32']['decile_edges']);expected_sh,shstat=EG.shuffle_gate(low,st,51515)
check('asset_38856_views_complete_and_calibration',N==9714 and len(idx)==4*N and np.array_equal(idx,np.repeat(np.arange(N),4)) and np.array_equal(rot,np.tile(np.arange(4),N)) and np.array_equal(cm,np.isin(idx,cal_ids)) and cm.sum()==12288 and cal_sha==man['calibration']['index_sha256_16'])
check('theta_and_strict_gate_recomputed',man['theta_q']==float(np.median(qv[cm])) and np.array_equal(low,(qv<man['theta_q']).astype(np.int8)) and EG.gate_low_q([1,2,3],2).tolist()==[1,0,0],theta=man['theta_q'],all_active=float(low.mean()),calibration_active=float(low[cm].mean()))
check('QES_all_strata_fixed_permutation',np.array_equal(shuf,expected_sh) and np.array_equal(dec,arrays['e_decile']) and all(low[st==s].sum()==shuf[st==s].sum() for s in np.unique(st)),strata=len(np.unique(st)),changed_fraction=float((shuf!=low).mean()))
train=ROOT/man['dataset']['train'];train_pan=ROOT/man['dataset']['train_pan']
check('actual_dataset_full_sha',EG.sha256_file(train)==man['dataset']['train_sha256'] and EG.sha256_file(train_pan)==man['dataset']['train_pan_sha256'])
T0,tman=Q.load_T0();th=state_hash(T0);cfg=cfgs[G.run_name('QE50',2026,arch=arch)];fa=cfg['train_feeder_args'];contract,_,_=Q.contract_from(cfg)
check('actual_T0_asset_and_margin',tman['file_sha256']=='16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32' and state_hash(T0.aligner)==man['teacher']['aligner_state_hash'] and T0.aligner_margin==4)
base=np.array([13,79,111,987]);rows_idx=(base[:,None]*4+np.arange(4)[None]).reshape(-1)
with h5py.File(train) as f,h5py.File(train_pan) as fp:rows=Q.read_rows(f,fp,base)
qt,ct,et=Q.compute_views(T0,4,rows,contract,2047.)
check('actual_CPU_online_vs_GPU_cache_16_views',np.max(abs(qt-qv[rows_idx]))<1e-5 and np.max(abs(et-arrays['e_roi32'][rows_idx]))<1e-5,max_abs_dq=float(np.max(abs(qt-qv[rows_idx]))),max_abs_de=float(np.max(abs(et-arrays['e_roi32'][rows_idx]))),label_flips=int((EG.gate_low_q(qt,man['theta_q'])!=low[rows_idx]).sum()))
v={k:EG.apply_view(EG.to_feeder_tensor(a,2047.),True,True,0) for k,a in rows.items()};gt,ms,lp,pan=(v[k] for k in ('gt','ms','lpan','pan'));meta=torch.tensor([[int(i),0,1,1] for i in base])
with torch.no_grad():
    mb=torch.nn.functional.interpolate(ms,scale_factor=4,mode='bicubic');c0=T0.predict_delta(pan,mb);res=[]
    for e in [[0,r] for r in (.25,-.25,.5,-.5,1,-1,2,-2)]+[[r,0] for r in (.25,-.25,.5,-.5,1,-1,2,-2)]:
        eps=torch.tensor(e)[None].expand(len(base),2);ce=T0.predict_delta(EG.warp_pan(pan,eps),mb);res.append((ce+eps-c0).abs().double())
    manual_q=torch.stack(res).mean((0,2));impl_q,_=EG.teacher_q(T0.aligner,pan,ms,4)
check('q_formula_independent_axis_sum',torch.allclose(manual_q,impl_q,atol=1e-7,rtol=0),max_diff=float((manual_q-impl_q).abs().max()),constant_response_q=EG.q_const_component())

# Actual trainer on W104, direct independently written total and A/U gradients.
tree=ast.parse((ROOT/'tools/pakd50_unit_tests.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_stubQ')
ns=dict(G=G,calQ=G.calibration(),KDVTrainer=KDVTrainer,resolve=resolve,types=types,T0=T0,GTAnchoredReconstructionKD=GTAnchoredReconstructionKD,torch=torch)
exec(compile(ast.Module(body=[node],type_ignores=[]),'unit_stub_only','exec'),ns);make=ns['_stubQ'];cal=G.calibration()
u=build(cfg);randomize_zero_params(u);S=PAModel(u,copy.deepcopy(T0.aligner).requires_grad_(True),aligner_margin=4,sampler=True)
def load_gate(mode,**kw):return EdgeGate.load(dict(mode=mode,asset=G.QEDGE9_CUE_ASSET,**kw),teacher=T0,train_h5=str(train),feeder_args=fa,sha_fn=lambda _:man['dataset']['train_sha256'])
state0=(torch.get_rng_state().clone(),np.random.get_state(),random.getstate());egl=load_gate('low_q');egs=load_gate('shuffle',perm_seed=51515)
check('cache_loading_preserves_training_RNG',torch.equal(state0[0],torch.get_rng_state()) and np.array_equal(state0[1][1],np.random.get_state()[1]) and state0[2]==random.getstate())
for case,eg in [('QE50',egl),('QES',egs),('QEC',EdgeGate.synthetic('const',c_E=.37))]:
    tr=make(case,copy.deepcopy(S).train(),eg);loss,info=tr._step(gt,ms,lp,pan,1,meta=meta);y,yt=info['y'],info['y_t'];es=(y-gt).abs().mean(1);et=(yt-gt).abs().mean(1)
    d=(et/(et+cal['tau_R'])).detach();a=((es-et).clamp_min(0)/(es+1e-6)).detach();lh=((1+d)*es).mean();lk=(.1*(1-d)*a*(y-yt).abs().mean(1)).mean()
    ei=output_edge_loss_per_sample(y,gt);weights=(eg.gate_for(meta,gt.device) if eg.needs_meta else torch.full((len(base),),eg.c_E));edge=cal['lambda_E']*(weights*ei).mean();expected=lh+lk+edge+1e-4*info['loss_off']
    ap=list(tr.M.aligner.parameters());up=list(tr.M.backbone.parameters())
    ga=torch.autograd.grad(expected,ap,retain_graph=True);gu=torch.autograd.grad(expected,up,retain_graph=True);offu=torch.autograd.grad(info['loss_off'],up,retain_graph=True,allow_unused=True)
    ge=torch.autograd.grad(info['_edge_w_t'],y,retain_graph=True)[0];loss.backward()
    ae=max(float((p.grad-g).abs().max()) for p,g in zip(ap,ga));ue=max(float((p.grad-g).abs().max()) for p,g in zip(up,gu))
    check(case+'_actual_manual_loss_gradients',abs(float(loss-expected))<1e-7 and ae<2e-6 and ue<2e-6 and all(g is None or not g.any() for g in offu) and (not (weights==0).any() or not ge[weights==0].any()) and all(p.grad is None for p in T0.parameters()),loss_error=abs(float(loss-expected)),A_error=ae,U_error=ue,gate=weights.tolist(),hard=float(lh),soft=float(lk))
    check(case+'_native_input_and_teacher_immutable',torch.equal(info['pan_view'],pan) and state_hash(T0)==th and info['eq_exercise']==1.)
    del tr,loss,info,expected,ga,gu,offu,ge

# Negative asset tests: all writes are temporary. Acceptance is a reproduced gap.
with tempfile.TemporaryDirectory(prefix='qedge9-asset-negative-') as td:
    p=pathlib.Path(td)
    def mutated(name,m=None,ar=None):
        mm=copy.deepcopy(man) if m is None else m;aa=arrays if ar is None else ar;npz=p/(name+'.npz');np.savez_compressed(npz,**aa);mm['npz']=str(npz);mm['npz_sha256']=EG.sha256_file(npz);jp=p/(name+'.json');jp.write_text(json.dumps(mm));return str(jp)
    def accepted(asset,mode='low_q',teacher=T0,**kw):
        try:return EdgeGate.load(dict(mode=mode,asset=asset,**kw),teacher=teacher,train_h5=str(train),feeder_args=fa,sha_fn=lambda _:man['dataset']['train_sha256'])
        except (ValueError,KeyError,FileNotFoundError):return None
    m=copy.deepcopy(man);m['theta_q']=None
    bug('missing_theta_accepted_by_trainer_loader',accepted(mutated('no-theta',m)) is not None)
    m=copy.deepcopy(man);m['theta_q']=1e9;a9=mutated('inconsistent-theta',m);e9=accepted(a9)
    bug('theta_disagrees_with_gate_table_accepted',e9 is not None and not torch.equal(e9.table,torch.ones_like(e9.table)))
    m=copy.deepcopy(man);m['bank']['id']='WRONG_BANK';m['bank']['K']=4
    bug('wrong_probe_bank_accepted',accepted(mutated('wrong-bank',m)) is not None)
    other=copy.deepcopy(T0);other.aligner_margin=0
    bug('changed_Teacher_view_margin_accepted',accepted(G.QEDGE9_CUE_ASSET,teacher=other) is not None,actual_margin=other.aligner_margin,cache_margin=man['teacher']['aligner_view_margin'])
    m=copy.deepcopy(man);m['qes']['seed']=7;aa={k:v.copy() for k,v in arrays.items()};aa['gate_shuffle']=EG.shuffle_gate(low,st,7)[0]
    bug('QES_wrong_permutation_seed_accepted',accepted(mutated('wrong-seed',m,aa),mode='shuffle',perm_seed=51515) is not None)
    ce=p/'wrong-pilot.json';ce.write_text(json.dumps(dict(c_E=.6,cue_npz_sha256=man['npz_sha256'],pilot_run='WRONG_W104_D123_S777',pilot_tag='best_hqnr',pilot_step=1010,pilot_width=104,pilot_depth=[1,2,3])))
    bug('QEC_wrong_pilot_identity_accepted',accepted(G.QEDGE9_CUE_ASSET,mode='const',c_E_file=str(ce)) is not None)
    aa={k:v.copy() for k,v in arrays.items()};aa['gate_low_q']=1-aa['gate_low_q'];aa['gate_shuffle']=1-aa['gate_shuffle'];an=mutated('changed-cue',ar=aa)
    eold=accepted(G.QEDGE9_CUE_ASSET);enew=accepted(an)
    bug('same_run_cfg_no_expected_cue_hash_rejects_replacement',eold is not None and enew is not None and torch.equal(eold.table,1-enew.table),note='loader has no expected hash from run/checkpoint; resume re-loads this path before loading accelerator state')
    # Actual verify command logic, corrupted theta with unchanged q/e: flips are
    # recorded but absent from pass_. Patch only recomputation to exact cached values.
    args=types.SimpleNamespace(asset=a9,config=Q.DEFAULT_CONFIG,n=4,seed=0,tol=1e-4)
    vr=np.sort(np.random.RandomState(0).choice(N,4,replace=False));ri=(vr[:,None]*4+np.arange(4)).reshape(-1);buf=io.StringIO()
    with patch.object(Q,'CAMP',str(p/'verify')),patch.object(Q,'load_T0',return_value=(T0,tman)),patch.object(Q,'sha_cached',return_value=man['dataset']['train_sha256']),patch.object(Q,'compute_views',return_value=(qv[ri],arrays['c0'][ri],arrays['e_roi32'][ri]+.1)),contextlib.redirect_stdout(buf):
        try:Q.verify(args)
        except SystemExit as e:rc=e.code
    vv=json.loads((p/'verify'/('verify_'+Q.server()+'.json')).read_text());(OUT/'verify_negative_evidence.json').write_text(json.dumps(vv,indent=2));bug('cue_verify_passes_label_flips_and_large_e_mismatch',rc==0 and vv['label_flips']>0 and vv['max_abs_de']>.09,verify=vv)

# The current exact-resume claim includes the next batch. Model/global RNG alone
# cannot restore a partially consumed RandomSampler iterator.
torch.manual_seed(2026);dl=torch.utils.data.DataLoader(torch.arange(128),batch_size=8,shuffle=True,num_workers=0);it=iter(dl);first=next(it);saved=torch.get_rng_state().clone();expected=next(it)
torch.set_rng_state(saved);resumed=next(iter(torch.utils.data.DataLoader(torch.arange(128),batch_size=8,shuffle=True,num_workers=0)))
bug('resume_global_RNG_does_not_restore_next_batch',not torch.equal(expected,resumed),expected=expected.tolist(),resumed=resumed.tolist(),note='same DataLoader construction as main; main does not save sampler cursor/worker augmentation state')

# Switch dry-run writes and reservation ledger source, with all operational
# state under /tmp and no scheduler/cron/process mutation invoked.
for srv in ('s5','s4'):
    with tempfile.TemporaryDirectory(prefix='qedge9-switch-'+srv+'-') as td:
        p=pathlib.Path(td);fixture(p,srv);(p/'tools/pakd50_unit_tests.py').write_text("print('ALL OK (separately verified)')\n");(p/'tools/qedge9_cue.py').write_text("print('cue verification stub: actual cache independently checked')\n")
        before={str(x.relative_to(p)):x.read_bytes() for x in (p/'work_dir').rglob('*') if x.is_file()}
        rr=subprocess.run(['bash','tools/qedge9_switch.sh','--dry-run'],cwd=p,env={**os.environ,'PYTHON':sys.executable},capture_output=True,text=True,timeout=60);(OUT/('switch_'+srv+'.log')).write_text(rr.stdout+rr.stderr)
        check('switch_'+srv+'_fixture_reaches_end',rr.returncode==0,rc=rr.returncode)
        mandatory=(p/G.QEDGE9_MANDATORY_FILE).read_text().splitlines();runs=[x for x in mandatory if x and not x.startswith('#')]
        check('switch_'+srv+'_QEDGE9_mandatory_complete',len(runs)==(6 if srv=='s5' else 3),runs=runs)
        bug('switch_'+srv+'_dry_run_mutates_live_control_files',(p/G.MANDATORY_FILE).exists() and (p/G.QEDGE9_MANDATORY_FILE).exists() and (p/G.RESERVATION_FILE).exists(),written=[G.MANDATORY_FILE,G.QEDGE9_MANDATORY_FILE,G.RESERVATION_FILE])
        if srv=='s5':
            led={'entries':{G.run_name('QE50',2026,arch=arch):dict(kind='run',status='FINISHED',hours=2.75)}};(p/'work_dir/_qedge9_budget').mkdir(exist_ok=True);(p/G.QEDGE9_LEDGER).write_text(json.dumps(led))
            with patch.object(G,'ROOT',str(p)):
                got=G.measured_hours_from_ledger(srv);correct=G.measured_hours_from_ledger(srv,led);ref=G.reservation_for(srv,'QE50@'+arch,got)
            bug('QEDGE9_measured_runtime_ignored_old_ledger_read',not got and correct.get('QE50@'+arch)==2.75 and ref['reference_train_h']==1.5,measured_default=got,own_ledger_measurement=correct,reservation=ref)

todo,dropped=G.schedule(True,lambda _:False,-24,lambda _:100,priority=G.priority_for('s5'),exempt=lambda x:G.branch_for('s5',x)=='QEDGE9',blocked=lambda _:False)
check('QEDGE9_admission_survives_expired_parent_deadline',todo==G.priority_for('s5') and not dropped)
todo,_=G.schedule(True,lambda _:False,100,lambda _:1,priority=['QE50@'+arch,'QEC@'+arch,'QES@'+arch],exempt=lambda _:True,blocked=lambda x:G.case_of(x)=='QEC')
bug('QEC_wait_is_skipped_in_favor_of_QES',list(map(G.case_of,todo))==['QE50','QES'],order=todo)
todo,_=G.schedule(True,lambda _:False,100,lambda _:1,priority=['QEC@'+arch],exempt=lambda _:True,blocked=lambda _:True)
source=(ROOT/'tools/_run_cases.sh').read_text();watch=(ROOT/'tools/_watchdog.sh').read_text()
bug('only_pending_cue_emits_empty_and_runner_DONE',not todo and 'if [ ${#EXTRA[@]} -eq 0 ]' in source and '추가 실행 없음"; break' in source and '[cases] DONE' in source,note='runner exits on empty gate pass; watchdog treats DONE as terminal; --pilot does not restart it')
bug('soft_time_switch_installs_72h_hard_start_deadline','--hours 72' in (ROOT/'tools/qedge9_switch.sh').read_text() and 'if past_deadline; then' in source)

prov=json.loads((OUT/'provenance.json').read_text());diff=[f['path'] for f in prov['files'] if hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest()!=f['sha256']]
check('frozen_source_unchanged',not diff,changed=diff)
R['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');R['summary']={k:dict(total=sum(x['kind']==k for x in R['checks']),passed=sum(x['kind']==k and x['passed'] for x in R['checks'])) for k in sorted({x['kind'] for x in R['checks']})}
(OUT/'verification.json').write_text(json.dumps(R,ensure_ascii=False,indent=2));print(json.dumps(R['summary']),flush=True)
sys.exit(0 if all(x['passed'] for x in R['checks']) else 1)
